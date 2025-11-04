# model.py
import zlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- U-Net blocks ----------------
def C2d(in_ch, out_ch, k=3, s=1, p=1, bias=True, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetSREncoder(nn.Module):
    """FP32 Encoder + Bottleneck"""
    def __init__(self, in_ch=3, base=64, use_bn=True):
        super().__init__()
        self.e1_1 = C2d(in_ch, base, use_bn=use_bn)
        self.e1_2 = C2d(base, base, use_bn=use_bn)
        self.pool1 = nn.MaxPool2d(2)

        self.e2_1 = C2d(base, base*2, use_bn=use_bn)
        self.e2_2 = C2d(base*2, base*2, use_bn=use_bn)
        self.pool2 = nn.MaxPool2d(2)

        self.e3_1 = C2d(base*2, base*4, use_bn=use_bn)
        self.e3_2 = C2d(base*4, base*4, use_bn=use_bn)
        self.pool3 = nn.MaxPool2d(2)

        self.e4_1 = C2d(base*4, base*8, use_bn=use_bn)
        self.e4_2 = C2d(base*8, base*8, use_bn=use_bn)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = C2d(base*8, base*16, use_bn=use_bn)

    @staticmethod
    def _match(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        e1_1 = self.e1_1(x)
        e1_2 = self.e1_2(e1_1)
        p1   = self.pool1(e1_2)

        e2_1 = self.e2_1(p1)
        e2_2 = self.e2_2(e2_1)
        p2   = self.pool2(e2_2)

        e3_1 = self.e3_1(p2)
        e3_2 = self.e3_2(e3_1)
        p3   = self.pool3(e3_2)

        e4_1 = self.e4_1(p3)
        e4_2 = self.e4_2(e4_1)
        p4   = self.pool4(e4_2)

        b = self.bottleneck(p4)
        return b, (e4_2, e3_2, e2_2, e1_2)


class UNetSRDecoderInt8Friendly(nn.Module):
    """
    INT8-friendly Decoder (QAT 대상)
    - ConvTranspose 대신 Upsample(nearest)+Conv
    - head: Conv -> PixelShuffle(scale)
    """
    def __init__(self, out_ch=3, base=64, scale=2, use_bn=True):
        super().__init__()
        assert scale in (2, 4)
        self.scale = scale

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d4_2 = C2d(base*16 + base*8, base*8, use_bn=use_bn)
        self.d4_1 = C2d(base*8, base*4, use_bn=use_bn)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d3_2 = C2d(base*4 + base*4, base*4, use_bn=use_bn)
        self.d3_1 = C2d(base*4, base*2, use_bn=use_bn)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d2_2 = C2d(base*2 + base*2, base*2, use_bn=use_bn)
        self.d2_1 = C2d(base*2, base, use_bn=use_bn)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d1_2 = C2d(base + base, base, use_bn=use_bn)
        self.d1_1 = C2d(base, base, use_bn=use_bn)

        self.head = nn.Sequential(
            nn.Conv2d(base, out_ch * (scale ** 2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(scale)
        )

    @staticmethod
    def _match(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, b, s4, s3, s2, s1):
        u4 = self.up4(b);  u4 = self._match(u4, s4)
        d4 = self.d4_2(torch.cat([u4, s4], dim=1))
        d4 = self.d4_1(d4)

        u3 = self.up3(d4); u3 = self._match(u3, s3)
        d3 = self.d3_2(torch.cat([u3, s3], dim=1))
        d3 = self.d3_1(d3)

        u2 = self.up2(d3); u2 = self._match(u2, s2)
        d2 = self.d2_2(torch.cat([u2, s2], dim=1))
        d2 = self.d2_1(d2)

        u1 = self.up1(d2); u1 = self._match(u1, s1)
        d1 = self.d1_2(torch.cat([u1, s1], dim=1))
        d1 = self.d1_1(d1)

        sr = self.head(d1)
        return sr


# --------------- Skip/Bottleneck compression (simulating comm) ---------------
class Int8Compressor:
    """Per-tensor affine int8 quantization + zlib 압축(바이트 수 측정)"""
    @staticmethod
    def quantize_per_tensor(x: torch.Tensor):
        x_cpu = x.detach().float().cpu()
        x_min = float(x_cpu.min())
        x_max = float(x_cpu.max())
        if x_max == x_min:
            scale, zp = 1.0, 0
            q = torch.zeros_like(x_cpu, dtype=torch.int8)
        else:
            qmin, qmax = -128, 127
            scale = (x_max - x_min) / (qmax - qmin)
            zp = int(round(qmin - x_min / scale))
            q = torch.clamp((x_cpu / scale + zp).round(), qmin, qmax).to(torch.int8)
        return q, float(scale), int(zp)

    @staticmethod
    def dequantize_per_tensor(q: torch.Tensor, scale: float, zp: int):
        return (q.float() - zp) * scale

    @staticmethod
    def compress_bytes(q: torch.Tensor) -> bytes:
        raw = q.cpu().numpy().tobytes()
        return zlib.compress(raw, level=6)

    @staticmethod
    def qdq_and_size(x: torch.Tensor):
        q, s, z = Int8Compressor.quantize_per_tensor(x)
        payload = Int8Compressor.compress_bytes(q)
        x_hat = Int8Compressor.dequantize_per_tensor(q, s, z).to(x.device)
        return x_hat, len(payload)


# --------------- Train-time wrapper (Encoder FP32 + Decoder QAT) ---------------
class SplitUNetSR_QATWrapper(nn.Module):
    """
    - encoder: FP32
    - decoder: QAT(FX) 준비된 모듈 (학습 중 fake-quant)
    - 중간 텐서(b, skips)는 int8 압축/복원으로 통신 손실 시뮬레이션
    """
    def __init__(self, encoder_fp32: UNetSREncoder, decoder_qat: nn.Module, simulate_comm=True):
        super().__init__()
        self.encoder = encoder_fp32
        self.decoder = decoder_qat
        self.sim_comm = simulate_comm
        self.register_buffer("avg_bytes_per_image", torch.zeros(1))
        self._n_images = 0

    def forward(self, x):
        b, (s4, s3, s2, s1) = self.encoder(x)
        bytes_total = 0
        if self.sim_comm:
            b,  sz_b  = Int8Compressor.qdq_and_size(b)
            s4, sz_s4 = Int8Compressor.qdq_and_size(s4)
            s3, sz_s3 = Int8Compressor.qdq_and_size(s3)
            s2, sz_s2 = Int8Compressor.qdq_and_size(s2)
            s1, sz_s1 = Int8Compressor.qdq_and_size(s1)
            bytes_total = sz_b + sz_s4 + sz_s3 + sz_s2 + sz_s1
            with torch.no_grad():
                self._n_images += x.shape[0]
                self.avg_bytes_per_image[0] = (self.avg_bytes_per_image[0]*(self._n_images-1) + bytes_total/x.shape[0]) / self._n_images
        sr = self.decoder(b, s4, s3, s2, s1)
        return sr


__all__ = [
    "UNetSREncoder",
    "UNetSRDecoderInt8Friendly",
    "SplitUNetSR_QATWrapper",
    "Int8Compressor",
]
