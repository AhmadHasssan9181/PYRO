"""
PYRO 🔥 Training Script v2

Loss     : Tversky (α=0.7, miss-fire penalty) + Focal (γ=2.0, hard-pixel focus)
Optimizer: AdamW with linear warmup + CosineAnnealingLR
Precision: AMP (automatic mixed precision)

Usage:
    python pyro_train.py
    python pyro_train.py --epochs 80 --batch 8 --img_size 384

Data layout (DATA_ROOT):
    <DATA_ROOT>/Images/*.jpg   — RGB fire images from FLAME dataset
    <DATA_ROOT>/Masks/*.png    — binary fire masks (pixel 1=fire, 0=background)
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from pyro_model import PyroNet, PyroLoss
from pyro_dataset import PyroDataset, make_split_datasets
from utils import compute_confusion_matrix, compute_miou

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = os.environ.get(
    "PYRO_DATA_ROOT",
    r"H:\FLAME_data\segmentation"
)

USE_SPLIT_FOLDERS = False

NUM_CLASSES  = 2
IMG_SIZE     = 512     # ↑ 384→512: captures thin fire lines missed at lower res
BATCH        = 6       # smaller batch for 512px VRAM budget
EPOCHS       = 150     # 3 cosine warm-restart cycles × 50 epochs
LR           = 3e-4
WARMUP_EPOCHS= 5       # linear LR warmup before first cosine cycle
WEIGHT_DECAY = 1e-4
FIRE_WEIGHT  = 60.0    # focal loss weight; slightly higher at 512px for tighter crops
IGNORE_INDEX = 255
SAVE_DIR     = "checkpoints"
USE_AMP      = True
RESUME_CKPT = r"checkpoints\best_pyro_fire.pth"  # set to "" to train from scratch
EARLY_STOP_PATIENCE = 20   # stop if FireIoU doesn't improve for this many epochs
NUM_WORKERS  = 0       # Windows: keep 0, data is RAM-cached anyway

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train PYRO fire segmentation model")
    p.add_argument("--epochs",   type=int,   default=EPOCHS)
    p.add_argument("--batch",    type=int,   default=BATCH)
    p.add_argument("--img_size", type=int,   default=IMG_SIZE)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--data",     type=str,   default=DATA_ROOT)
    return p.parse_args()


def get_loaders(data_root: str, img_size: int, batch: int):
    if USE_SPLIT_FOLDERS:
        train_ds = PyroDataset(
            images_dir=os.path.join(data_root, "train", "Images"),
            masks_dir=os.path.join(data_root, "train", "Masks"),
            img_size=img_size, is_val=False,
        )
        val_ds = PyroDataset(
            images_dir=os.path.join(data_root, "val", "Images"),
            masks_dir=os.path.join(data_root, "val", "Masks"),
            img_size=img_size, is_val=True,
        )
    else:
        train_ds, val_ds = make_split_datasets(
            images_dir=os.path.join(data_root, "Images"),
            masks_dir=os.path.join(data_root, "Masks"),
            img_size=img_size,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_fire_score = 0.0
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        seg_logits, fire_score = model(imgs)
        preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
        labs  = masks.cpu().numpy()
        hist += compute_confusion_matrix(preds, labs, NUM_CLASSES, ignore_index=IGNORE_INDEX)
        total_fire_score += fire_score.sum().item()
        n += imgs.size(0)
    miou, per_class = compute_miou(hist)
    avg_fire_score = total_fire_score / max(n, 1)
    return miou, per_class, avg_fire_score


def get_warmup_lr(epoch, warmup_epochs, base_lr):
    """Linear LR warmup — avoids early large gradients killing pretrained features."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def train():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"🔥 PYRO v2 — Fire Segmentation Training")
    print(f"   Device   : {DEVICE}")
    print(f"   Data root: {args.data}")
    print(f"   Img size : {args.img_size}px  |  Batch: {args.batch}  |  Epochs: {args.epochs}")
    print(f"   Loss     : Tversky(α=0.7,β=0.3) + Focal(γ=2, fire_w={FIRE_WEIGHT})")
    print(f"   LR warmup: {WARMUP_EPOCHS} epochs → CosineWarmRestarts (T0=50, ×3 cycles)")
    print(f"   Resume   : {RESUME_CKPT or 'scratch'}")
    print("─" * 60)

    train_loader, val_loader = get_loaders(args.data, args.img_size, args.batch)

    model     = PyroNet(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = PyroLoss(fire_weight=FIRE_WEIGHT, ignore_index=IGNORE_INDEX)

    # Separate LR for encoder (fine-tune slowly) vs decoder (learn fast)
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for p in model.parameters() if
                      not any(p is ep for ep in encoder_params)]
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},   # 10× slower encoder
        {'params': decoder_params, 'lr': args.lr},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine Warm Restarts — 3 cycles of 50 epochs each (escapes local minima)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    # ── Resume from best previous checkpoint if available ─────────────────────
    best_miou     = 0.0
    best_fire_iou = 0.0
    if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"   ✅ Resumed weights from {RESUME_CKPT}")
        if missing:
            print(f"      (new layers: {len(missing)} keys will train from scratch)")

    # ── Early stopping state ───────────────────────────────────────────────────
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):

        # ── LR Warmup ────────────────────────────────────────────────────────
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = get_warmup_lr(epoch - 1, WARMUP_EPOCHS, args.lr)
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = warmup_lr * (0.1 if i == 0 else 1.0)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{args.epochs}", ncols=110)
        running_loss  = 0.0
        running_tv    = 0.0
        running_focal = 0.0

        for imgs, masks in pbar:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=USE_AMP):
                seg_logits, _ = model(imgs)
                loss, tv_l, fc_l = criterion(seg_logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss  += loss.item()
            running_tv    += tv_l.item()
            running_focal += fc_l.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'tv': f'{tv_l.item():.4f}',
                'focal': f'{fc_l.item():.4f}',
            })

        # Advance warm-restart scheduler only after warmup
        if epoch > WARMUP_EPOCHS:
            scheduler.step(epoch - WARMUP_EPOCHS)

        avg_loss = running_loss / len(train_loader)
        cur_lr   = optimizer.param_groups[1]['lr']

        # Save latest every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(SAVE_DIR, "latest_pyro.pth"))

        # Validation
        miou, per_class, avg_fire_score = evaluate(model, val_loader)
        fire_iou = per_class[1] if len(per_class) > 1 else 0.0
        bg_iou   = per_class[0] if len(per_class) > 0 else 0.0

        print(
            f"  Ep {epoch:>3} | Loss {avg_loss:.4f} | "
            f"mIoU {miou:.4f} | FireIoU {fire_iou:.4f} | "
            f"BgIoU {bg_iou:.4f} | FireScore {avg_fire_score:.3f} | "
            f"LR {cur_lr:.2e}"
        )

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_pyro.pth"))
            print(f"  🏆 New best mIoU: {best_miou:.4f} → saved best_pyro.pth")

        if fire_iou > best_fire_iou:
            best_fire_iou = fire_iou
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_pyro_fire.pth"))
            print(f"  🔥 New best FireIoU: {best_fire_iou:.4f} → saved best_pyro_fire.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"\n  ⏹  Early stop: FireIoU flat for {EARLY_STOP_PATIENCE} epochs.")
                break

    print("─" * 60)
    print(f"✅ Training complete. Best mIoU: {best_miou:.4f}  |  Best FireIoU: {best_fire_iou:.4f}")
    print(f"   Checkpoints: ./{SAVE_DIR}/")


if __name__ == "__main__":
    train()
