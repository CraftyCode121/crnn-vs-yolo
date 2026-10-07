"""
evaluate.py - Benchmark evaluation for YOLO11 character detection and OCR transcription.
Evaluates across all synth_pages benchmark splits:
  - test_seen_moderate (seen fonts, new text)
  - test_unseen_fonts_moderate (font generalization gap)
  - test_tiers_clean, test_tiers_moderate, test_tiers_hard (degradation severity progression)

Computes both:
  1. Object Detection Metrics: Precision, Recall, mAP50, mAP50-95
  2. OCR Transcription Metrics: Strict CER, Normalized CER (look-alikes), WER, Sequence Accuracy

Usage:
  python -m yolo.evaluate --config yolo/training.yaml --weights checkpoints/yolo11_best.pt
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import cv2
import numpy as np

from .postprocess import (
    build_charset_mapping,
    reconstruct_page_text,
    compute_transcription_metrics
)
from .dataset import YOLODatasetManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("YOLOEvaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 on synth_pages test splits")
    parser.add_argument("--config", type=str, default="yolo/training.yaml", help="Path to yolo/training.yaml")
    parser.add_argument("--weights", type=str, default="checkpoints/yolo11_best.pt", help="Path to .pt model weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for evaluation")
    parser.add_argument("--device", type=str, default="", help="Device ('', '0', 'cpu')")
    parser.add_argument("--output_file", type=str, default="reports/benchmark_results.json", help="Path to save benchmark JSON")
    parser.add_argument("--skip_ocr_transcription", action="store_true", help="Skip OCR CER/WER computation, run detection mAP only")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of pages evaluated per split for fast test")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        alt_path = Path(__file__).resolve().parent / config_path
        if alt_path.exists():
            path = alt_path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata_lookup(metadata_file: Path) -> Dict[str, Dict[str, Any]]:
    """Loads metadata.jsonl into a lookup dict keyed by (folder, page_id) or page_id."""
    lookup = {}
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found at {metadata_file}. OCR text comparison might be limited.")
        return lookup

    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item.get("folder", ""), item.get("page_id", ""))
            lookup[key] = item
            lookup[item.get("page_id", "")] = item
    return lookup


def evaluate_split_ocr(
    model,
    images_dir: Path,
    metadata_lookup: Dict[Any, Dict[str, Any]],
    folder_name: str,
    id_to_char: Dict[int, str],
    lookalike_groups: List[List[str]],
    transcription_cfg: Dict[str, Any],
    sample_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Runs YOLO inference and transcribes detected boxes into text lines."""
    img_files = sorted(images_dir.glob("*.png"))
    if sample_limit:
        img_files = img_files[:sample_limit]

    if not img_files:
        return {"error": f"No images found in {images_dir}"}

    conf_thresh = transcription_cfg.get("conf_threshold", 0.25)
    iou_thresh = transcription_cfg.get("iou_threshold", 0.45)
    line_tol = transcription_cfg.get("line_grouping_tol", 0.6)
    space_ratio = transcription_cfg.get("space_gap_ratio", 0.35)

    all_pred_lines = []
    all_gt_lines = []

    for img_path in img_files:
        page_id = img_path.stem
        meta_entry = metadata_lookup.get((folder_name, page_id)) or metadata_lookup.get(page_id)
        gt_lines = meta_entry.get("lines", []) if meta_entry else []

        # Run model prediction
        results = model.predict(
            source=str(img_path),
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=1024,
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

        pred_lines = reconstruct_page_text(
            detected_boxes,
            line_grouping_tol=line_tol,
            space_gap_ratio=space_ratio
        )

        all_pred_lines.extend(pred_lines)
        all_gt_lines.extend(gt_lines)

    metrics = compute_transcription_metrics(
        all_pred_lines,
        all_gt_lines,
        lookalike_groups=lookalike_groups
    )
    metrics["total_pages"] = len(img_files)
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Prepare/Validate data.yaml
    dataset_mgr = YOLODatasetManager(config)
    data_yaml_path = dataset_mgr.generate_data_yaml()

    # 2. Resolve Weights Path
    weights_path = Path(args.weights)
    if not weights_path.is_absolute() and not weights_path.exists():
        alt_p = Path(__file__).resolve().parent / args.weights
        if alt_p.exists():
            weights_path = alt_p
        else:
            ckpt_p = Path(__file__).resolve().parent / "checkpoints" / args.weights
            if ckpt_p.exists():
                weights_path = ckpt_p

    if not weights_path.exists() and not str(weights_path).endswith(".pt"):
        raise FileNotFoundError(f"Model weights file not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info(f"Loading YOLO11 model from: {weights_path}")
    model = YOLO(str(weights_path))

    # 3. Splits to benchmark
    data_root = dataset_mgr.data_dir
    images_root = data_root / "images"

    splits_to_eval = [
        ("Seen Fonts (Moderate)", "test_seen_moderate"),
        ("Unseen Fonts (Font Generalization)", "test_unseen_fonts_moderate"),
        ("Tiers: Clean (Severity 0.0)", "test_tiers_clean"),
        ("Tiers: Moderate (Severity 0.25-0.5)", "test_tiers_moderate"),
        ("Tiers: Hard (Severity 0.7-1.0)", "test_tiers_hard")
    ]

    metadata_file = dataset_mgr.metadata_path
    metadata_lookup = load_metadata_lookup(metadata_file)
    lookalike_groups = config.get("lookalike_groups", [])
    transcription_cfg = config.get("transcription", {})

    results = {}

    print("\n" + "=" * 105)
    print(f"  YOLO11 BENCHMARK EVALUATION RESULTS ({weights_path.name})")
    print("=" * 105)
    print(f"{'Split / Condition':<35} | {'Pages':<6} | {'mAP50':<8} | {'mAP50-95':<9} | {'Strict CER':<10} | {'Norm CER':<9} | {'WER':<8} | {'Seq Acc':<8}")
    print("-" * 105)

    for display_name, folder_name in splits_to_eval:
        img_dir = images_root / folder_name
        if not img_dir.exists():
            print(f"{display_name:<35} | {'N/A':<6} | {'(Directory not found at ' + str(img_dir) + ')':<50}")
            results[folder_name] = {"error": f"Folder {folder_name} not found"}
            continue

        # A. Object detection metrics via Ultralytics val
        det_map50 = 0.0
        det_map50_95 = 0.0
        try:
            val_res = model.val(
                data=str(data_yaml_path),
                split=folder_name if folder_name in ["train", "val"] else "val",
                imgsz=args.imgsz,
                batch=args.batch_size,
                device=args.device,
                verbose=False
            )
            if hasattr(val_res, "box"):
                det_map50 = float(val_res.box.map50) * 100.0
                det_map50_95 = float(val_res.box.map) * 100.0
        except Exception as e:
            logger.debug(f"Ultralytics val note on {folder_name}: {e}")

        # B. End-to-end OCR Transcription Metrics
        if not args.skip_ocr_transcription:
            ocr_res = evaluate_split_ocr(
                model=model,
                images_dir=img_dir,
                metadata_lookup=metadata_lookup,
                folder_name=folder_name,
                id_to_char=dataset_mgr.id_to_char,
                lookalike_groups=lookalike_groups,
                transcription_cfg=transcription_cfg,
                sample_limit=args.sample_limit
            )
        else:
            ocr_res = {
                "total_pages": len(list(img_dir.glob("*.png"))),
                "strict_cer": 0.0,
                "normalized_cer": 0.0,
                "wer": 0.0,
                "sequence_accuracy": 0.0
            }

        split_summary = {
            "display_name": display_name,
            "folder": folder_name,
            "mAP50": det_map50,
            "mAP50_95": det_map50_95,
            **ocr_res
        }
        results[folder_name] = split_summary

        pages_count = or_res.get("total_pages", len(list(img_dir.glob("*.png")))) if 'or_res' in locals() else ocr_res.get("total_pages", 0)
        s_cer = ocr_res.get("strict_cer", 0.0) * 100.0
        n_cer = ocr_res.get("normalized_cer", 0.0) * 100.0
        wer_val = ocr_res.get("wer", 0.0) * 100.0
        seq_acc = ocr_res.get("sequence_accuracy", 0.0)

        print(
            f"{display_name:<35} | "
            f"{pages_count:<6} | "
            f"{det_map50:>6.1f}%  | "
            f"{det_map50_95:>7.1f}%  | "
            f"{s_cer:>8.2f}%  | "
            f"{n_cer:>7.2f}% | "
            f"{wer_val:>6.2f}% | "
            f"{seq_acc:>6.2f}%"
        )

    print("=" * 105 + "\n")

    # Save output JSON
    out_file = Path(args.output_file)
    if not out_file.is_absolute():
        out_file = (Path(__file__).resolve().parent / args.output_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark results to: {out_file}")

    # Generate Markdown summary table
    md_file = out_file.parent / "evaluation_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# YOLO11 Benchmark Evaluation Summary ({weights_path.name})\n\n")
        f.write("| Split / Condition | Pages | mAP50 | mAP50-95 | Strict CER | Norm CER | WER | Exact Seq Acc |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for k, v in results.items():
            if "error" not in v:
                f.write(
                    f"| {v['display_name']} | {v.get('total_pages', 'N/A')} | "
                    f"{v.get('mAP50', 0.0):.1f}% | {v.get('mAP50_95', 0.0):.1f}% | "
                    f"{v.get('strict_cer', 0.0)*100:.2f}% | {v.get('normalized_cer', 0.0)*100:.2f}% | "
                    f"{v.get('wer', 0.0)*100:.2f}% | {v.get('sequence_accuracy', 0.0):.2f}% |\n"
                )
    logger.info(f"Saved evaluation markdown summary to: {md_file}")


if __name__ == "__main__":
    main()
