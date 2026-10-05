"""
infer.py - Fast inference script for CRNN OCR model.
Transcribes text line crops (PNG images) using trained model weights.
Supports:
  - Single line crop transcription
  - Directory batch transcription
  - Greedy or Beam Search CTC decoding
  - Confidence scoring

Usage:
  python infer.py --weights checkpoints/crnn_best_model.weights.h5 --image sample_line.png
  python infer.py --weights checkpoints/crnn_best_model.weights.h5 --image_dir ../synth_pages/output/crnn/test_tiers_hard
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf

from .dataset import CharsetVocab, preprocess_image
from .model import build_crnn_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("CRNNInfer")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CRNN OCR inference")
    parser.add_argument("--config", type=str, default="training.yaml", help="Path to training.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .weights.h5 file")
    parser.add_argument("--image", type=str, default=None, help="Path to single text line crop image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to directory of text line crop images")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "beam_search"], help="CTC decoding method")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        alt_path = Path(__file__).resolve().parent / config_path
        if alt_path.exists():
            path = alt_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_single_image(model, vocab, image_path: str, config: dict, method: str = "greedy", beam_width: int = 10):
    img_h = config["dataset"].get("img_height", 48)
    img_w = config["dataset"].get("max_img_width", 800)
    channels = config["dataset"].get("channels", 1)
    normalize = config["dataset"].get("normalize", True)
    pad_val = float(config["dataset"].get("pad_value", 255.0))
    downsample = 4

    img_bytes = tf.io.read_file(image_path)
    img_tensor, valid_w = preprocess_image(
        img_bytes,
        target_height=img_h,
        target_width=img_w,
        channels=channels,
        normalize=normalize,
        pad_value=pad_val
    )

    batch_img = tf.expand_dims(img_tensor, 0)
    valid_steps = tf.maximum(1, valid_w // downsample)
    input_length = tf.expand_dims(valid_steps, 0)

    logits = model(batch_img, training=False)
    decoded_sparse, log_prob = model.decode(
        logits,
        input_length,
        method=method,
        beam_width=beam_width
    )
    decoded_dense = tf.sparse.to_dense(
        decoded_sparse,
        default_value=vocab.blank_idx
    ).numpy()

    indices = decoded_dense[0].tolist()
    text = vocab.indices_to_text(indices, remove_blank=True)
    confidence = float(np.exp(log_prob.numpy()[0])) if method == "greedy" else float(np.exp(-log_prob.numpy()[0][0]))

    return text, confidence


def main():
    args = parse_args()
    config = load_config(args.config)
    vocab = CharsetVocab(config)

    model = build_crnn_model(config, vocab.num_classes, vocab.blank_idx)
    model.load_weights(args.weights)

    if args.image:
        text, conf = predict_single_image(model, vocab, args.image, config, args.method, args.beam_width)
        print(f"\nImage: {args.image}")
        print(f"Recognized Text: {text}")
        print(f"Confidence:      {conf:.4f}\n")

    elif args.image_dir:
        img_dir = Path(args.image_dir)
        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        print(f"\nProcessing {len(img_files)} images from {img_dir}...\n")
        print(f"{'Filename':<35} | {'Transcription'}")
        print("-" * 80)
        for p in img_files[:20]:
            text, _ = predict_single_image(model, vocab, str(p), config, args.method, args.beam_width)
            print(f"{p.name:<35} | {text}")
        print("-" * 80)
    else:
        print("Please provide either --image <path> or --image_dir <path>")


if __name__ == "__main__":
    main()
