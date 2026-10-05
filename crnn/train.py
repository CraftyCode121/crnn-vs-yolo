"""
train.py - Main training script for CRNN OCR model using TensorFlow.
Usage:
  python train.py --config training.yaml
  python train.py --config training.yaml --epochs 50 --batch_size 32
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import tensorflow as tf

from .dataset import CRNNDatasetBuilder
from .model import build_crnn_model
from .callbacks import build_callbacks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CRNNTrain")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN OCR model with TensorFlow and CTC Loss")
    parser.add_argument(
        "--config",
        type=str,
        default="training.yaml",
        help="Path to YAML training configuration file"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data_root", type=str, default=None, help="Override path to output/crnn folder")
    parser.add_argument("--labels_file", type=str, default=None, help="Override path to labels.tsv")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch dry-run on 2 batches for pipeline verification")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        # Try relative to this script's directory
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

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.data_root is not None:
        config["paths"]["data_root"] = args.data_root
    if args.labels_file is not None:
        config["paths"]["labels_file"] = args.labels_file

    logger.info("Initializing CRNN Dataset Builder...")
    dataset_builder = CRNNDatasetBuilder(config)
    vocab = dataset_builder.vocab

    splits_cfg = config.get("splits", {})
    train_split_name = splits_cfg.get("train", "train")
    val_split_name = splits_cfg.get("val", "val")

    batch_size = config["training"].get("batch_size", 32)
    epochs = config["training"].get("epochs", 60)

    limit = 64 if args.dry_run else None

    logger.info(f"Building training dataset from split '{train_split_name}'...")
    train_ds = dataset_builder.create_tf_dataset(
        split_folder=train_split_name,
        batch_size=batch_size,
        is_training=True,
        limit_samples=limit
    )

    logger.info(f"Building validation dataset from split '{val_split_name}'...")
    val_ds = dataset_builder.create_tf_dataset(
        split_folder=val_split_name,
        batch_size=batch_size,
        is_training=False,
        limit_samples=limit
    )

    logger.info("Constructing CRNN Model Architecture...")
    model = build_crnn_model(
        config=config,
        num_classes=vocab.num_classes,
        blank_idx=vocab.blank_idx
    )
    model.summary(print_fn=logger.info)

    # Optimizer setup
    lr = float(config["training"].get("learning_rate", 0.001))
    clipnorm = float(config["training"].get("clipnorm", 5.0))
    opt_type = config["training"].get("optimizer", "adam").lower()

    if opt_type == "adamw":
        wd = float(config["training"].get("weight_decay", 1e-4))
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=clipnorm)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

    model.compile(optimizer=optimizer)

    # Assemble callbacks: EarlyStopping, ModelCheckpoint, EpochAccuracyReportCallback, etc.
    callbacks = build_callbacks(config, val_ds, vocab)

    if args.dry_run:
        logger.info("Executing dry run (1 epoch)...")
        epochs = 1

    logger.info(f"Starting model training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model weights
    ckpt_dir = Path(config["paths"].get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_weights_path = ckpt_dir / "crnn_final_model.weights.h5"
    model.save_weights(str(final_weights_path))
    logger.info(f"Saved final model weights to {final_weights_path}")
    logger.info("Training complete. Review generated reports in reports/ directory.")


if __name__ == "__main__":
    main()
