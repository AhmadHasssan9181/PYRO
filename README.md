# PYRO 🔥 — Pixel-level pYRotechnic Outlining

**Fire Segmentation Model for UAV Aerial Imagery**

PYRO is a lightweight, mobile-deployable neural network for real-time fire segmentation from autonomous drones. It is optimized for the [FLAME Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs).

## Architecture

The model uses a highly efficient, mobile-friendly architecture:
- **Encoder:** MobileNetV3-Small (pretrained on ImageNet)
- **Bottleneck:** ASPP (Atrous Spatial Pyramid Pooling) for multi-scale fire context (captures both small embers and large blazes)
- **Decoder:** 5-stage UNet with SE (Squeeze-and-Excitation) attention on skip connections to suppress background clutter, followed by Depthwise-Separable convolutions.

Outputs a full 384×384 resolution segmentation mask.

## Files

- `pyro_model.py`: Core architecture definitions (MobileNetV3 Encoder, ASPP, Decoder, Losses).
- `pyro_train.py`: Training script for the model on the FLAME dataset.
- `pyro_dataset.py`: Dataloading and preprocessing pipeline.
- `pyro_inference.py`: Inference script for generating masks from UAV images.
- `utils.py`: Helper functions for metrics and visualization.
- `checkpoints/best_pyro_fire.pth`: The best trained checkpoint ready for deployment.

## Loss Functions

Handles extreme class imbalance (fire pixels < 1% of the image) using a combination of:
- **Tversky Loss**: Heavily penalizes false negatives (missed fire).
- **Focal Loss**: Forces the model to focus on hard-to-classify fire pixels instead of easy background.

## Usage

### Inference
You can run inference using the pre-trained weights (`best_pyro_fire.pth`) on a test image:
```bash
python pyro_inference.py --image path/to/drone_image.jpg --weights checkpoints/best_pyro_fire.pth
```

### Training
To train the model from scratch on the FLAME dataset:
```bash
python pyro_train.py --data_dir /path/to/FLAME --epochs 150 --batch_size 16
```
