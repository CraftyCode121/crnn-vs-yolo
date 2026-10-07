"""
infer.py - Inference and text line transcription on document images using YOLO11.
Detects 2D character bounding boxes on a page, groups them into lines, inserts
spaces, and prints reconstructed text in reading order.

Usage:
  python -m yolo.infer --image path/to/sample_page.png --weights checkpoints/yolo11_best.pt
  python -m yolo.infer --image path/to/sample_page.png --save_vis
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import cv2
import numpy as np

from .postprocess import build_charset_mapping, reconstruct_page_text, group_boxes_into_lines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("YOLOInfer")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11 OCR inference on a page image")
    parser.add_argument("--image", type=str, required=True, help="Path to input page image (.png, .jpg)")
    parser.add_argument("--weights", type=str, default="checkpoints/yolo11_best.pt", help="Path to model weights (.pt)")
    parser.add_argument("--config", type=str, default="yolo/training.yaml", help="Path to yolo/training.yaml")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--save_vis", action="store_true", help="Save annotated visualization image")
    parser.add_argument("--output_vis", type=str, default="reports/preview_prediction.png", help="Path for visualization output")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        alt_path = Path(__file__).resolve().parent / config_path
        if alt_path.exists():
            path = alt_path

    if not path.exists():
        # Fallback default
        return {"charset": {
            "lowercase": "abcdefghijklmnopqrstuvwxyz",
            "uppercase": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "digits": "0123456789",
            "punctuation": [".", ",", ";", ":", "!", "?", "'", "\"", "-", "(", ")"]
        }}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_visualizations(
    img_bgr: np.ndarray,
    lines_of_boxes: list,
    output_path: Path
):
    """Draws color-coded character boxes and labels on the image."""
    vis = img_bgr.copy()
    colors = [
        (255, 60, 60), (60, 220, 60), (60, 100, 255),
        (255, 180, 40), (220, 60, 220), (40, 220, 220)
    ]

    for line_idx, line in enumerate(lines_of_boxes):
        color = colors[line_idx % len(colors)]
        for box in line:
            x0, y0, x1, y1 = int(box["x0"]), int(box["y0"]), int(box["x1"]), int(box["y1"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
            # Draw tiny label above box
            cv2.putText(
                vis, box["char"], (x0, max(12, y0 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    logger.info(f"Saved OCR bounding box visualization to: {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    charset_cfg = config.get("charset", {})
    _, id_to_char = build_charset_mapping(charset_cfg)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {img_path}")

    # Resolve weights
    weights_p = Path(args.weights)
    if not weights_p.is_absolute() and not weights_p.exists():
        alt_p = Path(__file__).resolve().parent / args.weights
        if alt_p.exists():
            weights_p = alt_p
        else:
            ckpt_p = Path(__file__).resolve().parent / "checkpoints" / args.weights
            if ckpt_p.exists():
                weights_p = ckpt_p

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info(f"Loading YOLO11 model: {weights_p}")
    model = YOLO(str(weights_p))

    logger.info(f"Running inference on {img_path} (imgsz={args.imgsz}, conf={args.conf})...")
    results = model.predict(
        source=str(img_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False
    )

    detected_boxes = []
    if results and len(results) > 0:
        res = results[0]
        boxes_data = res.boxes
        if boxes_data is not None and len(boxes_data) > 0:
            xyxy = boxes_data.xyxy.cpu().numpy()
            cls = boxes_data.cls.cpu().numpy().astype(int)
            conf = boxes_data.conf.cpu().numpy()

            for i in range(len(xyxy)):
                c_id = cls[i]
                char_str = id_to_char.get(c_id, "")
                x0, y0, x1, y1 = xyxy[i]
                detected_boxes.append({
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "cx": float((x0 + x1) / 2.0),
                    "cy": float((y0 + y1) / 2.0),
                    "w": float(x1 - x0),
                    "h": float(y1 - y0),
                    "cls": int(c_id),
                    "conf": float(conf[i]),
                    "char": char_str
                })

    logger.info(f"Detected {len(detected_boxes)} characters.")

    transcription_cfg = config.get("transcription", {})
    line_tol = transcription_cfg.get("line_grouping_tol", 0.6)
    space_ratio = transcription_cfg.get("space_gap_ratio", 0.35)

    grouped_lines = group_boxes_into_lines(detected_boxes, line_grouping_tol=line_tol)
    reconstructed_lines = reconstruct_page_text(detected_boxes, line_grouping_tol=line_tol, space_gap_ratio=space_ratio)

    print("\n" + "=" * 70)
    print(f"  YOLO11 RECONSTRUCTED DOCUMENT TEXT ({img_path.name})")
    print("=" * 70)
    for idx, line in enumerate(reconstructed_lines, 1):
        print(f"  Line {idx:02d}: {line}")
    print("=" * 70 + "\n")

    if args.save_vis:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            out_vis = Path(args.output_vis)
            draw_visualizations(img_bgr, grouped_lines, out_vis)


if __name__ == "__main__":
    main()
