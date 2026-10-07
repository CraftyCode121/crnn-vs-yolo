"""
train.py - Fine-tune YOLO11 on synthetic document pages using Ultralytics.
Usage:
  python train.py --config training.yaml
  python train.py --config training.yaml --model_path yolo11n.pt
  python train.py --config training.yaml --model_path checkpoints/yolo11_best.pt --epochs 30
  python train.py --resume
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import yaml

from .dataset import YOLODatasetManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("YOLOTrain")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 character detector on synth_pages")
    parser.add_argument(
        "--config",
        type=str,
        default="training.yaml",
        help="Path to YAML training configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to existing model checkpoint (.pt) to load or resume from"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size (default: 1024)")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate (lr0)")
    parser.add_argument("--device", type=str, default=None, help="Override device ('', '0', '0,1', 'cpu')")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--data", type=str, default=None, help="Override path to data.yaml")
    parser.add_argument("--project", type=str, default=None, help="Override output project directory")
    parser.add_argument("--name", type=str, default=None, help="Override run experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint in run")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        alt_path = Path(__file__).resolve().parent / config_path
        if alt_path.exists():
            path = alt_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {path}")
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Dataset Management & data.yaml Generation
    logger.info("Initializing YOLO Dataset Manager...")
    dataset_mgr = YOLODatasetManager(config)
    data_yaml_path = dataset_mgr.generate_data_yaml()

    # 2. Determine Model Source
    default_model_name = config.get("model", {}).get("name", "yolo11n.pt")
    selected_model_path = default_model_name
    is_resuming = args.resume

    if args.model_path is not None:
        p = Path(args.model_path)
        if not p.is_absolute() and not p.exists():
            # Check relative to script dir or checkpoints_dir
            alt_p = Path(__file__).resolve().parent / args.model_path
            ckpt_p = Path(__file__).resolve().parent / "checkpoints" / args.model_path
            if alt_p.exists():
                p = alt_p
            elif ckpt_p.exists():
                p = ckpt_p

        if not p.exists() and not str(p).endswith(".pt"):
            # Could be a standard Ultralytics model alias like 'yolo11s.pt'
            selected_model_path = args.model_path
            logger.info(f"Using Ultralytics pretrained model: {selected_model_path}")
        elif p.exists():
            selected_model_path = str(p.resolve())
            logger.info(f"Validated existing checkpoint: {selected_model_path}")
        else:
            raise FileNotFoundError(f"Specified --model_path does not exist: {args.model_path}")

    # 3. Import Ultralytics with friendly error handling
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "Ultralytics is not installed. Please run: pip install ultralytics\n"
            "or: pip install -r yolo/requirements.txt"
        )
        sys.exit(1)

    # 4. Construct Ultralytics Model
    if is_resuming:
        logger.info(f"Resuming training run from checkpoint: {selected_model_path}")
        model = YOLO(selected_model_path)
    else:
        if args.model_path is not None:
            logger.info(f"Fine-tuning from existing model checkpoint: {selected_model_path}")
        else:
            logger.info(f"Initializing from pretrained YOLO11 base: {selected_model_path}")
        model = YOLO(selected_model_path)

    # 5. Training Arguments Preparation
    tr_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    epochs = args.epochs if args.epochs is not None else tr_cfg.get("epochs", 50)
    batch_size = args.batch_size if args.batch_size is not None else tr_cfg.get("batch_size", 16)
    imgsz = args.imgsz if args.imgsz is not None else tr_cfg.get("imgsz", 1024)
    lr0 = args.lr if args.lr is not None else float(tr_cfg.get("lr0", 0.01))
    device = args.device if args.device is not None else tr_cfg.get("device", "")
    workers = args.workers if args.workers is not None else tr_cfg.get("workers", 4)
    data_file = args.data if args.data is not None else str(data_yaml_path)
    project_dir = args.project if args.project is not None else str((Path(__file__).resolve().parent / paths_cfg.get("project", "./runs")).resolve())
    run_name = args.name if args.name is not None else paths_cfg.get("name", "yolo11_crnn_benchmark")

    train_kwargs = {
        "data": data_file,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "project": project_dir,
        "name": run_name,
        "exist_ok": True,
        "optimizer": tr_cfg.get("optimizer", "auto"),
        "lr0": lr0,
        "lrf": float(tr_cfg.get("lrf", 0.01)),
        "patience": tr_cfg.get("patience", 15),
        "save": tr_cfg.get("save", True),
        "plots": tr_cfg.get("plots", True),
        "val": tr_cfg.get("val", True),
        # Mandatory OCR Augmentation constraints (Pitfall 4):
        "fliplr": float(tr_cfg.get("fliplr", 0.0)),
        "flipud": float(tr_cfg.get("flipud", 0.0)),
        "degrees": float(tr_cfg.get("degrees", 0.0)),
        "mosaic": float(tr_cfg.get("mosaic", 0.0)),
        "mixup": float(tr_cfg.get("mixup", 0.0)),
        "scale": float(tr_cfg.get("scale", 0.2)),
        "translate": float(tr_cfg.get("translate", 0.1))
    }

    if is_resuming:
        train_kwargs["resume"] = True

    logger.info("=" * 70)
    logger.info("  STARTING YOLO11 FINE-TUNING ON SYNTH_PAGES")
    logger.info("=" * 70)
    logger.info(f"  Model       : {selected_model_path}")
    logger.info(f"  Data Config : {data_file}")
    logger.info(f"  Image Size  : {imgsz}x{imgsz} (native width preserved)")
    logger.info(f"  Epochs      : {epochs}")
    logger.info(f"  Batch Size  : {batch_size}")
    logger.info(f"  Augmentations: fliplr={train_kwargs['fliplr']}, mosaic={train_kwargs['mosaic']}")
    logger.info(f"  Outputs     : {project_dir}/{run_name}")
    logger.info("=" * 70)

    # 6. Execute Native Ultralytics Training
    results = model.train(**train_kwargs)

    # 7. Checkpointing & Exports
    ckpt_dir = (Path(__file__).resolve().parent / paths_cfg.get("checkpoints_dir", "./checkpoints")).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(project_dir) / run_name / "weights"
    if weights_dir.exists():
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        if best_pt.exists():
            shutil.copy(best_pt, ckpt_dir / "yolo11_best.pt")
            logger.info(f"Saved best model checkpoint to: {ckpt_dir / 'yolo11_best.pt'}")
        if last_pt.exists():
            shutil.copy(last_pt, ckpt_dir / "yolo11_last.pt")
            logger.info(f"Saved last model checkpoint to: {ckpt_dir / 'yolo11_last.pt'}")

    logger.info("Training complete. Review training curves and metrics in runs/ and reports/.")
    return results


if __name__ == "__main__":
    main()
