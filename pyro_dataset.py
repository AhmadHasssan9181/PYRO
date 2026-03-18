"""
PYRO 🔥 Dataset Loader — PyroDataset
Loads FLAME segmentation images and binary fire masks.

FLAME mask convention (confirmed by pixel probing):
  Pixel value == 1  → class 1 (FIRE)
  Pixel value == 0  → class 0 (BACKGROUND)
  Fire pixels are <1% of total pixels — extreme class imbalance.

Paths expected:
  images_dir  →  H:/FLAME_data/segmentation/Images/
  masks_dir   →  H:/FLAME_data/segmentation/Masks/
"""

import os
import cv2
import numpy as np
import pathlib
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class PyroDataset(Dataset):
    """
    Dataset for FLAME fire segmentation.

    Args:
        images_dir : path to folder of RGB fire/no-fire images (.jpg)
        masks_dir  : path to folder of binary mask images (.png)
                     Pass None for inference-only (no masks).
        img_size   : square resize target (default 384)
        is_val     : if True, no augmentation (only resize + normalize)
        cache_mode : load all images into RAM on init (fast training)
        ignore_index: value used to mark ignored pixels in masks (default 255)
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png'}

    def __init__(
        self,
        images_dir: str,
        masks_dir: str | None = None,
        img_size: int = 384,
        is_val: bool = False,
        cache_mode: bool = True,
        ignore_index: int = 255,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.ignore_index = ignore_index
        self.cache_mode = cache_mode

        if not is_val:
            p = str(images_dir).lower()
            is_val = 'val' in p or 'test' in p
        self.is_val = is_val

        self.transforms = self._build_transforms(img_size, is_val)

        # ── Collect paths (stem-based pairing: img.jpg ↔ mask.png) ──────────
        def collect_stems(folder):
            if folder and os.path.exists(folder):
                return {
                    p.stem: p
                    for p in pathlib.Path(folder).iterdir()
                    if p.suffix.lower() in self.IMG_EXTS
                }
            return {}

        img_map  = collect_stems(images_dir)
        mask_map = collect_stems(masks_dir) if masks_dir else {}

        if mask_map:
            common = sorted(img_map.keys() & mask_map.keys())
            if not common:
                self.image_paths = sorted(img_map.values())
                self.mask_paths  = sorted(mask_map.values())
            else:
                self.image_paths = [img_map[s]  for s in common]
                self.mask_paths  = [mask_map[s] for s in common]
        else:
            self.image_paths = sorted(img_map.values())
            self.mask_paths  = []

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {images_dir}")

        # ── RAM Cache ─────────────────────────────────────────────────────────
        self.cached_data = []
        if self.cache_mode:
            split = 'Val' if is_val else 'Train'
            print(f"⏳ [{split}] Caching {len(self.image_paths)} images → RAM @ {img_size}px ...")
            for i in tqdm(range(len(self.image_paths))):
                img, mask = self._load_and_resize(i)
                self.cached_data.append((img, mask))
            print(f"✅ Cache complete. {len(self.cached_data)} samples ready.")

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _build_transforms(self, img_size: int, is_val: bool) -> A.Compose:
        if is_val:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        return A.Compose([
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.5, 1.0),        # more aggressive crop to hit fire
                ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.Transpose(p=0.2),

            # Fire colour simulation
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
            A.CLAHE(p=0.2),             # local contrast — helps low-light fire

            # UAV motion/atmospheric
            A.MotionBlur(blur_limit=(3, 9), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.3),
            A.ImageCompression(quality_range=(60, 95), p=0.2),  # JPEG artefacts

            # Coarse dropout = simulates partial occlusion by smoke
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0, p=0.2,
            ),

            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_and_resize(self, idx: int):
        """Load image + mask, resize both to img_size, binarize mask."""
        img_path = str(self.image_paths[idx])
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        mask = None
        if idx < len(self.mask_paths):
            mask_path = str(self.mask_paths[idx])
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is not None:
                raw_mask = cv2.resize(raw_mask, (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_NEAREST)
                mask = self._binarize_mask(raw_mask)

        if mask is None:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        return img, mask

    @staticmethod
    def _binarize_mask(mask: np.ndarray) -> np.ndarray:
        """
        FLAME mask binarisation.

        FLAME masks store fire as pixel value 1 (NOT 255).
        Any non-zero pixel = fire class (1).
        Zero pixels = background (0).
        """
        return (mask > 0).astype(np.uint8)   # 0→0 (bg), 1→1 (fire)

    # ── Dataset Interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self.cache_mode:
            img, mask = self.cached_data[idx]
            if img is None:
                return self.__getitem__(0)
        else:
            img, mask = self._load_and_resize(idx)
            if img is None:
                return self.__getitem__(max(0, idx - 1))

        aug = self.transforms(image=img, mask=mask)
        return aug['image'], aug['mask'].long()   # [3,H,W] float, [H,W] long


# ── Utility: auto 80/20 split from a single flat folder ──────────────────────

def make_split_datasets(
    images_dir: str,
    masks_dir: str,
    img_size: int = 384,
    val_fraction: float = 0.2,
    cache_mode: bool = True,
    seed: int = 42,
):
    """
    Creates an 80/20 train/val split from a flat images+masks directory pair.
    Returns (train_ds, val_ds).
    """
    from torch.utils.data import Subset

    full_ds = PyroDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=img_size,
        is_val=False,
        cache_mode=cache_mode,
    )
    n = len(full_ds)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val

    train_ds = Subset(full_ds, indices[:n_train])
    val_ds   = Subset(full_ds, indices[n_train:])

    print(f"📊 Split: {n_train} train / {n_val} val  (seed={seed})")
    return train_ds, val_ds
