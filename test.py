# test.py
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    UNetSREncoder,
    UNetSRDecoderInt8Friendly,
    Int8Compressor,
)

from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def build_int8_decoder_skeleton(dec_fp, encoder, in_ch, patch, backend="fbgemm", device="cpu"):
    # FX graph needs example inputs with real shapes
    dummy = torch.randn(1, in_ch, patch, patch, device=device)
    b,(s4,s3,s2,s1) = encoder(dummy)
    qconfig = get_default_qat_qconfig(backend)
    dec_qat = prepare_qat_fx(dec_fp, {"": qconfig}, example_inputs=(b,s4,s3,s2,s1)).eval()
    dec_int8 = convert_fx(dec_qat)
    return dec_int8


@torch.no_grad()
def run_inference(ckpt_path, img_path, scale=2, in_ch=3, base=64, patch=128, show=False, simulate_comm=True):
    device = get_device()
    # skeletons
    enc = UNetSREncoder(in_ch=in_ch, base=base, use_bn=True).to(device).eval()
    dec_fp = UNetSRDecoderInt8Friendly(out_ch=in_ch, base=base, scale=scale, use_bn=True).to(device).eval()

    # load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder_fp32"])

    # build INT8 decoder and load weights
    dec_int8 = build_int8_decoder_skeleton(dec_fp, enc, in_ch, patch, device=device)
    dec_int8.load_state_dict(ckpt["decoder_int8"])
    dec_int8.eval()

    # image
    img = Image.open(img_path).convert("RGB") if in_ch==3 else Image.open(img_path).convert("L")
    W,H = img.size
    lr = img  # 이미 LR이라 가정. 필요 시 리사이즈/전처리 조정

    if in_ch==1:
        lr_t = torch.from_numpy(np.array(lr, dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0).to(device)
    else:
        lr_t = torch.from_numpy(np.array(lr, dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(device)

    # encoder (FP32)
    b,(s4,s3,s2,s1) = enc(lr_t)

    # optional comm simulation
    if simulate_comm:
        b  = Int8Compressor.qdq_and_size(b)[0]
        s4 = Int8Compressor.qdq_and_size(s4)[0]
        s3 = Int8Compressor.qdq_and_size(s3)[0]
        s2 = Int8Compressor.qdq_and_size(s2)[0]
        s1 = Int8Compressor.qdq_and_size(s1)[0]

    # decoder (INT8 GraphModule; takes float tensors, quantizes internally)
    sr = dec_int8(b, s4, s3, s2, s1)
    sr = torch.clamp(sr, 0.0, 1.0)

    # save result
    out_np = sr[0].cpu().permute(1,2,0).numpy()
    out_img = (out_np*255).astype("uint8")
    if in_ch == 1:
        Image.fromarray(out_img[...,0], mode="L").save("sr_out.png")
    else:
        Image.fromarray(out_img, mode="RGB").save("sr_out.png")
    print("Saved: sr_out.png")

    if show:
        try:
            import cv2
            bic = np.array(lr.resize((W*scale, H*scale), Image.BICUBIC))
            side = np.hstack([bic[:,:,::-1] if in_ch==3 else bic, out_img[:,:,::-1] if in_ch==3 else out_img])
            cv2.imshow(f"Bicubic vs UNetSR INT8 x{scale}", side)
            cv2.waitKey(0); cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   type=str, required=True, help="final_int8_decoder.pth")
    ap.add_argument("--img",    type=str, required=True)
    ap.add_argument("--in_ch",  type=int, default=3, choices=[1,3])
    ap.add_argument("--scale",  type=int, default=2, choices=[2,4])
    ap.add_argument("--base",   type=int, default=64)
    ap.add_argument("--patch",  type=int, default=128, help="example input patch for FX graph build")
    ap.add_argument("--show",   action="store_true")
    ap.add_argument("--no-sim-comm", action="store_true", help="disable comm simulation at test")
    args = ap.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        img_path=args.img,
        scale=args.scale,
        in_ch=args.in_ch,
        base=args.base,
        patch=args.patch,
        show=args.show,
        simulate_comm=(not args.no_sim_comm),
    )


if __name__ == "__main__":
    main()
