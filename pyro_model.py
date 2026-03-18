"""
PYRO 🔥 — Pixel-level pYRotechnic Outlining
Fire Segmentation Model for UAV Aerial Imagery (FLAME Dataset)

Architecture: MobileNetV3-Small encoder + ASPP bottleneck + 5-stage UNet decoder.

Design goals:
  - Mobile-deployable (companion phone computer on autonomous drone)
  - Full 384×384 output resolution
  - ASPP (Atrous Spatial Pyramid Pooling) at bottleneck for multi-scale fire context
  - SE (Squeeze-and-Excite) attention on skip connections to suppress background

MBv3-Small spatial strides (verified by probing at 384×384 input):
  Layer  0: stride  2 → [16,  H/2]   192×192
  Layer  3: stride  8 → [24,  H/8]    48×48
  Layer  8: stride 16 → [48, H/16]    24×24
  Layer 12: stride 32 → [576,H/32]    12×12  ← bottleneck → ASPP here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, dilation=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad*dilation,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DepthwiseSep(nn.Module):
    """Efficient depthwise-separable conv — mobile backbone of the decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = ConvBnRelu(in_ch, in_ch, k=3, pad=1, groups=in_ch)
        self.pw = ConvBnRelu(in_ch, out_ch, k=1, pad=0)
    def forward(self, x): return self.pw(self.dw(x))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Applied after skip-fusion to amplify fire-like channels.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x).view(x.size(0), -1, 1, 1)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures fire at multiple scales: small (nearby) and large (distant UAV view).
    Rates tuned for 12×12 feature maps (input 384, stride 32).
    """
    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        # 1×1 conv
        self.c1 = ConvBnRelu(in_ch, out_ch, k=1, pad=0)
        # dilated 3×3 convs — rates 2,4,6 work well at 12×12 spatial size
        self.c2 = ConvBnRelu(in_ch, out_ch, dilation=2)
        self.c3 = ConvBnRelu(in_ch, out_ch, dilation=4)
        self.c4 = ConvBnRelu(in_ch, out_ch, dilation=6)
        # global context branch — uses GroupNorm (works at 1×1, unlike BatchNorm)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.ReLU6(inplace=True),
        )
        # fuse all 5 branches
        self.proj = ConvBnRelu(out_ch * 5, out_ch, k=1, pad=0)
        self.drop = nn.Dropout2d(p=0.1)

    def forward(self, x):
        size = x.shape[2:]
        branches = [
            self.c1(x),
            self.c2(x),
            self.c3(x),
            self.c4(x),
            F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False),
        ]
        return self.drop(self.proj(torch.cat(branches, dim=1)))


class DecoderBlock(nn.Module):
    """
    Upsample  →  cat(skip)  →  SE attention  →  2× DepthwiseSep
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1)
        fused_ch = in_ch // 2 + skip_ch
        self.se  = SEBlock(fused_ch)
        self.conv = nn.Sequential(
            DepthwiseSep(fused_ch, out_ch),
            DepthwiseSep(out_ch,   out_ch),
        )

    def forward(self, x, skip=None):
        x = F.relu(self.up(x), inplace=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.se(x)
        return self.conv(x)


class FreeUpBlock(nn.Module):
    """Upsample with no skip (used where MBv3 has no matching feature map)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            DepthwiseSep(in_ch // 2, out_ch),
            DepthwiseSep(out_ch,     out_ch),
        )
    def forward(self, x):
        return self.conv(F.relu(self.up(x), inplace=True))


# ─────────────────────────────────────────────────────────────────────────────
# PyroNet
# ─────────────────────────────────────────────────────────────────────────────

class PyroNet(nn.Module):
    """
    PYRO Fire Segmentation Network v2

    Encoder  : MobileNetV3-Small (ImageNet pretrained) — mobile-grade, fast
    Bottlenck: ASPP — multi-scale fire context (small embers to large blaze)
    Decoder  : 5-stage UNet with SE-attention skip fusion + DepthwiseSep convs

    Outputs:
      seg_logits  [B, 2, H, W]  — fire / background logits at full resolution
      fire_score  [B]           — per-image mean fire probability (0→1)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        assert num_classes == 2, "PyroNet is a binary fire/no-fire segmentor."
        self.num_classes = num_classes

        weights = "IMAGENET1K_V1" if pretrained else None
        mbv3 = models.mobilenet_v3_small(weights=weights)
        self.encoder = mbv3.features        # 13 InvertedResidual stages
        self.skip_ids = {0, 3, 8}           # H/2(16ch), H/8(24ch), H/16(48ch)

        # ── Bottleneck: 576ch @ H/32 → ASPP → 128ch ──────────────────────────
        self.aspp = ASPP(in_ch=576, out_ch=128)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Stage 1: H/32 → H/16  +  skip[8]  (48 ch)
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=48, out_ch=96)

        # Stage 2: H/16 → H/8   +  skip[3]  (24 ch)
        self.dec2 = DecoderBlock(in_ch=96,  skip_ch=24, out_ch=64)

        # Stage 3: H/8  → H/4   (no skip at H/4 in MBv3-Small)
        self.dec3 = FreeUpBlock(in_ch=64,   out_ch=48)

        # Stage 4: H/4  → H/2   +  skip[0]  (16 ch)
        self.dec4 = DecoderBlock(in_ch=48,  skip_ch=16, out_ch=32)

        # Stage 5: H/2  → H     (no skip — full resolution clean-up)
        self.dec5 = FreeUpBlock(in_ch=32,   out_ch=24)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.seg_head = nn.Sequential(
            DepthwiseSep(24, 24),
            nn.Conv2d(24, num_classes, kernel_size=1),
        )

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Kaiming init for all decoder + head parameters (not encoder)."""
        decoder_mods = [
            self.aspp, self.dec1, self.dec2, self.dec3,
            self.dec4, self.dec5, self.seg_head,
        ]
        for module in decoder_mods:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        target_hw = x.shape[2:]
        skips = {}
        cur = x
        for i, layer in enumerate(self.encoder):
            cur = layer(cur)
            if i in self.skip_ids:
                skips[i] = cur

        # Bottleneck at H/32
        feat = self.aspp(cur)               # cur = output of last encoder layer (idx 12)

        feat = self.dec1(feat, skips[8])    # H/32 → H/16
        feat = self.dec2(feat, skips[3])    # H/16 → H/8
        feat = self.dec3(feat)              # H/8  → H/4
        feat = self.dec4(feat, skips[0])    # H/4  → H/2
        feat = self.dec5(feat)              # H/2  → H

        # Final size guard
        if feat.shape[2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode='bilinear', align_corners=False)

        seg_logits = self.seg_head(feat)                                    # [B,2,H,W]
        fire_score = torch.softmax(seg_logits, dim=1)[:, 1].mean(dim=[1, 2])  # [B]
        return seg_logits, fire_score


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky loss — generalisation of Dice that lets you penalise
    false negatives more than false positives.

    With alpha=0.7, beta=0.3: strongly penalises MISSED fire (FN).
    This is critical because fire pixels are <1% of all pixels.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha   # weight for false negatives (missing fire)
        self.beta   = beta    # weight for false positives (false alarms)
        self.smooth = smooth

    def forward(self, logits, targets):
        probs     = torch.softmax(logits, dim=1)[:, 1]      # P(fire) [B,H,W]
        fire_true = (targets == 1).float()

        TP = (probs * fire_true).sum(dim=[1, 2])
        FP = (probs * (1 - fire_true)).sum(dim=[1, 2])
        FN = ((1 - probs) * fire_true).sum(dim=[1, 2])

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return (1.0 - tversky).mean()


class FocalLoss(nn.Module):
    """
    Focal loss — down-weights easy background examples so the
    gradient is dominated by the rare fire pixels.
    """
    def __init__(self, gamma: float = 2.0, fire_weight: float = 50.0,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.register_buffer('weight', torch.tensor([1.0, fire_weight]))

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device),
            ignore_index=self.ignore_index,
            reduction='none',
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        # Only average over non-ignored pixels
        valid = (targets != self.ignore_index)
        return focal[valid].mean() if valid.any() else focal.mean()


class PyroLoss(nn.Module):
    """
    Combined loss for extreme class imbalance (fire < 1% of pixels):

      α · Tversky(α_fn=0.7)   — punishes missed fire hard
    + β · Focal(γ=2.0)        — focuses gradient on hard fire pixels
    + γ · BCE(fire_w=50)      — global class correction

    Total = 0.5 * Tversky + 0.5 * Focal
    """

    def __init__(self, fire_weight: float = 50.0, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)
        self.focal   = FocalLoss(gamma=2.0, fire_weight=fire_weight,
                                 ignore_index=ignore_index)

    def forward(self, logits, targets):
        tv_l  = self.tversky(logits, targets)
        fc_l  = self.focal(logits, targets)
        total = 0.5 * tv_l + 0.5 * fc_l
        return total, tv_l, fc_l
