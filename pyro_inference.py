"""
PYRO 🔥 Inference Script

Runs fire segmentation on a single image or a folder of images.
Produces a vivid fire-overlay visualization and prints fire coverage %.

Usage:
    # Single image
    python pyro_inference.py --image path\to\frame.jpg --model checkpoints\best_pyro.pth

    # Whole folder
    python pyro_inference.py --folder path\to\images\ --model checkpoints\best_pyro.pth

    # Use best-fire-iou checkpoint
    python pyro_inference.py --image frame.jpg --model checkpoints\best_pyro_fire.pth
"""

import os
import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pyro_model import PyroNet

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 384
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Fire overlay color (BGR) — electric cyan for max contrast vs orange fire + dark smoke
FIRE_COLOR = np.array([255, 220, 0], dtype=np.uint8)    # electric cyan
FIRE_ALPHA = 0.60   # overlay opacity for fire pixels
OUTPUT_SIZE = 640   # side-by-side output width per panel

# ── Preprocessing ─────────────────────────────────────────────────────────────
_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = _transform(image=img_rgb)['image'].unsqueeze(0)
    return tensor.to(DEVICE)


# ── Visualisation ─────────────────────────────────────────────────────────────

def build_overlay(orig_bgr: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Overlay fire pixels with a vivid red-orange tint.
    Non-fire regions remain as original with slight darkening.

    Returns a BGR image the same size as orig_bgr.
    """
    h, w = orig_bgr.shape[:2]

    # Upsample mask to original resolution
    mask_upscaled = cv2.resize(
        pred_mask.astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = orig_bgr.copy()
    fire_px = mask_upscaled == 1

    if fire_px.any():
        # Blend fire color onto fire pixels
        overlay[fire_px] = (
            FIRE_ALPHA * FIRE_COLOR +
            (1.0 - FIRE_ALPHA) * orig_bgr[fire_px].astype(np.float32)
        ).astype(np.uint8)

        # Add glowing edge effect: slight white hotspot on brightest fire areas
        fire_region = orig_bgr[fire_px].mean(axis=1)
        bright_mask = np.zeros_like(mask_upscaled, dtype=np.float32)
        bright_mask[fire_px] = np.clip(fire_region / 255.0, 0, 1)
        glow = (bright_mask[..., None] * np.array([255, 255, 180], dtype=np.float32)).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.9, glow, 0.1, 0)

    return overlay


def add_stats_bar(img: np.ndarray, fire_pct: float, fire_score: float) -> np.ndarray:
    """Add a semi-transparent stats bar at the bottom of image."""
    h, w = img.shape[:2]
    bar_h = 44
    bar = img.copy()

    # Dark semi-transparent bar
    bar[h - bar_h:h] = (bar[h - bar_h:h].astype(np.float32) * 0.4).astype(np.uint8)

    # Fire danger rating
    danger = "!! CRITICAL" if fire_pct > 30 else ("!  HIGH" if fire_pct > 10 else ("MEDIUM" if fire_pct > 1 else "CLEAR"))

    cv2.putText(
        bar,
        f"  Fire Coverage: {fire_pct:.1f}%   |   Peak Conf: {fire_score*100:.1f}%   |   {danger}",
        (10, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return bar


def label_panel(img: np.ndarray, label: str) -> np.ndarray:
    """Add a top-left label to a panel."""
    out = img.copy()
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(model: PyroNet, img_bgr: np.ndarray):
    """Returns (pred_mask [H,W], fire_prob_map [H,W], peak_conf float)."""
    tensor = preprocess(img_bgr)
    with torch.no_grad():
        seg_logits, _ = model(tensor)
    fire_probs = torch.softmax(seg_logits, dim=1)[:, 1].squeeze(0).cpu().numpy()  # [H,W]
    pred_mask  = (fire_probs > 0.5).astype(np.uint8)
    peak_conf  = float(fire_probs.max())   # highest confidence on any single pixel
    return pred_mask, fire_probs, peak_conf


def process_image(model: PyroNet, img_path: str, out_dir: str):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[WARN] Could not read: {img_path}")
        return

    pred_mask, fire_probs, peak_conf = run_inference(model, img_bgr)

    # Stats
    total_px = pred_mask.size
    fire_px  = pred_mask.sum()
    fire_pct = fire_px / total_px * 100.0

    # Build panels
    s = OUTPUT_SIZE
    orig_panel    = cv2.resize(img_bgr, (s, s))
    overlay_panel = cv2.resize(build_overlay(img_bgr, pred_mask), (s, s))

    orig_panel    = label_panel(orig_panel, "Original")
    overlay_panel = label_panel(overlay_panel, "PYRO Detection")

    combined = np.hstack([orig_panel, overlay_panel])
    combined = add_stats_bar(combined, fire_pct, peak_conf)

    # Save
    basename  = os.path.splitext(os.path.basename(img_path))[0]
    out_path  = os.path.join(out_dir, f"{basename}_pyro.jpg")
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 92])

    danger_str = "CRITICAL" if fire_pct > 30 else ("HIGH" if fire_pct > 10 else ("MEDIUM" if fire_pct > 1 else "CLEAR"))
    print(f"  [{danger_str:>8}]  {os.path.basename(img_path)}  |  "
          f"Coverage: {fire_pct:5.1f}%  |  PeakConf: {peak_conf*100:.1f}%  →  {out_path}")

    return fire_pct, peak_conf


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PYRO 🔥 Fire Segmentation Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--model",  type=str, default=r"checkpoints\best_pyro.pth",
                        help="Path to PyroNet checkpoint (.pth)")
    parser.add_argument("--out", type=str, default="pyro_results",
                        help="Output directory for result images")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"🔥 PYRO — Fire Detection Inference")
    print(f"   Model  : {args.model}")
    print(f"   Device : {DEVICE}")
    print(f"   Output : {args.out}/")
    print("─" * 60)

    # Load model
    model = PyroNet(num_classes=2, pretrained=False).to(DEVICE)
    ckpt  = torch.load(args.model, map_location=DEVICE)
    # Support both raw state_dict and wrapped checkpoint
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"✅ Model loaded from {args.model}\n")

    # Collect images
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    if args.image:
        images = [args.image]
    else:
        images = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        print(f"Found {len(images)} images in {args.folder}\n")

    for img_path in images:
        process_image(model, img_path, args.out)

    print("─" * 60)
    print(f"✅ Done. Results in: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
