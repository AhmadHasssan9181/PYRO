import os
import argparse
import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from pyro_model import PyroNet
from pyro_inference import (
    DEVICE, 
    OUTPUT_SIZE, 
    run_inference, 
    build_overlay, 
    add_stats_bar, 
    label_panel
)

def process_video(model: PyroNet, video_path: str, out_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    s = OUTPUT_SIZE
    out_w = s * 2
    out_h = s

    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{basename}_pyro.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    print(f"Processing video: {os.path.basename(video_path)} ({total_frames} frames @ {fps:.1f} fps)")
    
    pbar = tqdm(total=total_frames, desc="Inferencing") if tqdm else None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the current frame
        pred_mask, fire_probs, peak_conf = run_inference(model, frame)
        total_px = pred_mask.size
        fire_px  = pred_mask.sum()
        fire_pct = fire_px / total_px * 100.0

        # Build side-by-side visualization
        orig_panel    = cv2.resize(frame, (s, s))
        overlay_panel = cv2.resize(build_overlay(frame, pred_mask), (s, s))

        orig_panel    = label_panel(orig_panel, "Original")
        overlay_panel = label_panel(overlay_panel, "PYRO Detection")

        combined = np.hstack([orig_panel, overlay_panel])
        combined = add_stats_bar(combined, fire_pct, peak_conf)

        writer.write(combined)
        frame_count += 1
        
        if pbar:
            pbar.update(1)
        elif frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    if pbar:
        pbar.close()

    cap.release()
    writer.release()
    print(f"✅ Video saved to: {os.path.abspath(out_path)}")


def main():
    parser = argparse.ArgumentParser(description="PYRO 🔥 Video Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default=r"checkpoints\best_pyro.pth", help="Path to PyroNet checkpoint (.pth)")
    parser.add_argument("--out", type=str, default="pyro_results", help="Output directory for result video")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"🔥 PYRO — Video Inference")
    print(f"   Video  : {args.video}")
    print(f"   Model  : {args.model}")
    print(f"   Device : {DEVICE}")
    print(f"   Output : {args.out}/")
    print("─" * 60)

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found at {args.model}")
        return

    # Load model
    model = PyroNet(num_classes=2, pretrained=False).to(DEVICE)
    ckpt = torch.load(args.model, map_location=DEVICE)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    
    print(f"✅ Model loaded from {args.model}\n")

    process_video(model, args.video, args.out)
    print("─" * 60)


if __name__ == "__main__":
    main()
