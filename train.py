# train.py
import os, glob, random, time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== Your model imports =====
# If you have models/sr_model.py with __init__.py re-export:
from models import (
    UNetSREncoder,
    UNetSRDecoderInt8Friendly,
    SplitUNetSR_QATWrapper,
)

# QAT (FX)
from torch.ao.quantization import get_default_qat_qconfig
from torch.ao.quantization.qconfig import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx


# ---------- simple logger ----------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- Dataset: HR -> LR on-the-fly ----------
class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=2, patch_size=128, in_ch=3, train=True):
        super().__init__()
        self.paths = sorted([
            p for p in glob.glob(str(Path(hr_dir) / "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.scale = scale
        self.patch = patch_size
        self.in_ch = in_ch
        self.train = train

    def __len__(self):
        return len(self.paths)

    def _load_img(self, p):
        img = Image.open(p).convert("RGB")
        if self.in_ch == 1:
            img = img.convert("L")
        return img

    def __getitem__(self, idx):
        img = self._load_img(self.paths[idx])
        W, H = img.size
        s = self.scale
        ph = self.patch * s
        pw = self.patch * s

        if self.train:
            if W < pw or H < ph:
                img = img.resize((max(pw, W), max(ph, H)), Image.BICUBIC)
                W, H = img.size
            x0 = random.randint(0, W - pw)
            y0 = random.randint(0, H - ph)
            hr = img.crop((x0, y0, x0 + pw, y0 + ph))
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            if W >= pw and H >= ph:
                x0 = (W - pw) // 2
                y0 = (H - ph) // 2
                hr = img.crop((x0, y0, x0 + pw, y0 + ph))
            else:
                hr = img.resize((pw, ph), Image.BICUBIC)

        lr = hr.resize((self.patch, self.patch), Image.BICUBIC)

        if self.in_ch == 1:
            hr = torch.from_numpy(np.array(hr, dtype=np.float32) / 255.).unsqueeze(0)
            lr = torch.from_numpy(np.array(lr, dtype=np.float32) / 255.).unsqueeze(0)
        else:
            hr = torch.from_numpy(np.array(hr, dtype=np.float32) / 255.).permute(2, 0, 1)
            lr = torch.from_numpy(np.array(lr, dtype=np.float32) / 255.).permute(2, 0, 1)
        return lr, hr


# ---------- Loss ----------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


# ---------- Helpers ----------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_models(in_ch=3, base=64, scale=2, device="cuda"):
    enc = UNetSREncoder(in_ch=in_ch, base=base, use_bn=True).to(device)
    dec = UNetSRDecoderInt8Friendly(out_ch=in_ch, base=base, scale=scale, use_bn=True).to(device)
    return enc, dec


@torch.no_grad()
def get_example_inputs(encoder, in_ch, patch, device):
    dummy = torch.randn(1, in_ch, patch, patch, device=device)
    b, (s4, s3, s2, s1) = encoder(dummy)
    return b, s4, s3, s2, s1


def prepare_decoder_qat(decoder: nn.Module, example_inputs, backend="fbgemm"):
    qconfig = get_default_qat_qconfig(backend)
    qmap = QConfigMapping().set_global(qconfig)
    return prepare_qat_fx(decoder, qmap, example_inputs=example_inputs)


# ---------- Train ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hr_dir", type=str, required=True)
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints_unetsr_qat")
    parser.add_argument("--in_ch", type=int, default=3, choices=[1, 3])
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--qat_backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])

    # logging/data-loader controls
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0부터 시작 권장)")
    parser.add_argument("--pin_memory", action="store_true", help="DataLoader pin_memory 사용")
    parser.add_argument("--log_every", type=int, default=50, help="미니배치 로그 주기")

    args = parser.parse_args()

    device = get_device()
    torch.backends.cudnn.benchmark = True

    # data
    train_ds = SRDataset(args.train_hr_dir, scale=args.scale, patch_size=args.patch, in_ch=args.in_ch, train=True)
    val_ds = SRDataset(args.val_hr_dir, scale=args.scale, patch_size=args.patch, in_ch=args.in_ch, train=False)
    train_dl = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_memory, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_memory
    )

    log(f"Device: {device}")
    log(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")
    log(f"Dataloader -> workers={args.workers}, pin_memory={args.pin_memory}, bs={args.bs}")

    # models
    enc, dec = build_models(in_ch=args.in_ch, base=args.base, scale=args.scale, device=device)
    b, s4, s3, s2, s1 = get_example_inputs(enc, args.in_ch, args.patch, device)
    log(f"Example shapes -> b={tuple(b.shape)}, s4={tuple(s4.shape)}, s3={tuple(s3.shape)}, s2={tuple(s2.shape)}, s1={tuple(s1.shape)}")

    dec_qat = prepare_decoder_qat(dec, (b, s4, s3, s2, s1), backend=args.qat_backend).to(device)
    log("QAT graph prepared.")

    model = SplitUNetSR_QATWrapper(enc, dec_qat, simulate_comm=True).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = CharbonnierLoss()

    best_val = 1e9
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        log(f"Epoch {epoch} starting... (iters={len(train_dl)})")
        model.train()
        ep_loss = 0.0
        t_epoch = time.time()

        for i, (lr_img, hr_img) in enumerate(train_dl):
            if i == 0:
                log("First batch fetched.")

            t0 = time.time()
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += loss.item() * lr_img.size(0)

            if (i % args.log_every) == 0:
                dt = (time.time() - t0) * 1000.0
                log(f"  it {i:5d}/{len(train_dl):5d} | loss={loss.item():.4f} | {dt:.1f} ms/batch")

        sched.step()
        train_loss = ep_loss / len(train_dl.dataset)

        # eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_dl:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                sr = model(lr_img)
                val_loss += criterion(sr, hr_img).item()
        val_loss /= max(1, len(val_dl))

        avg_kb = float(model.avg_bytes_per_image.item()) / 1024.0
        log(f"Epoch {epoch} done in {(time.time() - t_epoch):.1f}s | train={train_loss:.4f} | val={val_loss:.4f} | comm~{avg_kb:.1f} KB/img")

        # save QAT-float checkpoint
        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pth")
        torch.save({
            "encoder_fp32": model.encoder.state_dict(),
            "decoder_qat_float": model.decoder.state_dict(),
            "args": vars(args)
        }, ckpt_path)
        log(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder_fp32": model.encoder.state_dict(),
                "decoder_qat_float": model.decoder.state_dict(),
                "args": vars(args)
            }, os.path.join(args.out_dir, "best_qat_float.pth"))
            log("Saved: best_qat_float.pth (improved)")

    # Convert decoder to real INT8
    log("Converting decoder to INT8 ...")
    dec_int8 = convert_fx(model.decoder.eval())
    torch.save({
        "encoder_fp32": model.encoder.state_dict(),
        "decoder_int8": dec_int8.state_dict(),
        "args": vars(args)
    }, os.path.join(args.out_dir, "final_int8_decoder.pth"))
    log("Saved: final_int8_decoder.pth")


if __name__ == "__main__":
    main()
