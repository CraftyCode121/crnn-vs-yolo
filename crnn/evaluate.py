"""
evaluate.py - Benchmark evaluation script for CRNN OCR model.
Evaluates across all synth_pages test splits:
  - test_seen_moderate (seen fonts, new text)
  - test_unseen_fonts_moderate (font generalization)
  - test_tiers_clean, test_tiers_moderate, test_tiers_hard (degradation severity progression)

Outputs detailed Strict CER, Normalized CER (lookalikes), WER, and exact sequence accuracy.
Usage:
  python evaluate.py --config training.yaml --weights checkpoints/crnn_best_model.weights.h5
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import yaml
import tensorflow as tf

from .dataset import CRNNDatasetBuilder
from .model import build_crnn_model
from .metrics import compute_epoch_metrics, normalize_lookalikes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CRNNEvaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CRNN OCR checkpoint on synth_pages splits")
    parser.add_argument("--config", type=str, default="training.yaml", help="Path to training.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to .weights.h5 file")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--output_file", type=str, default="reports/benchmark_results.json", help="Path to save benchmark JSON")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        alt_path = Path(__file__).resolve().parent / config_path
        if alt_path.exists():
            path = alt_path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_split(
    model,
    dataset_builder: CRNNDatasetBuilder,
    split_name: str,
    batch_size: int,
    lookalike_groups: list
) -> dict:
    vocab = dataset_builder.vocab
    try:
        ds = dataset_builder.create_tf_dataset(
            split_folder=split_name,
            batch_size=batch_size,
            is_training=False
        )
    except Exception as e:
        logger.warning(f"Could not build dataset for '{split_name}': {e}")
        return {"error": str(e)}

    predictions = []
    ground_truths = []

    for batch in ds:
        inputs, _ = batch
        images = inputs["image"]
        labels = inputs["label"].numpy()
        input_length = inputs["input_length"]
        label_length = inputs["label_length"].numpy()

        logits = model(images, training=False)
        decoded_sparse, _ = model.decode(logits, input_length, method="greedy")
        decoded_dense = tf.sparse.to_dense(
            decoded_sparse,
            default_value=vocab.blank_idx
        ).numpy()

        batch_size_actual = images.shape[0]
        for i in range(batch_size_actual):
            pred_indices = decoded_dense[i].tolist()
            pred_text = vocab.indices_to_text(pred_indices, remove_blank=True)

            gt_len = int(label_length[i])
            gt_indices = labels[i, :gt_len].tolist()
            gt_text = vocab.indices_to_text(gt_indices, remove_blank=True)

            predictions.append(pred_text)
            ground_truths.append(gt_text)

    if not ground_truths:
        return {"total_lines": 0}

    metrics = compute_epoch_metrics(predictions, ground_truths, lookalike_groups=lookalike_groups)
    metrics["total_lines"] = len(ground_truths)
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_builder = CRNNDatasetBuilder(config)
    vocab = dataset_builder.vocab

    logger.info("Initializing CRNN Model...")
    model = build_crnn_model(config, vocab.num_classes, vocab.blank_idx)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Check relative to checkpoints
        alt_path = Path(config["paths"]["checkpoint_dir"]) / args.weights
        if alt_path.exists():
            weights_path = alt_path

    logger.info(f"Loading weights from {weights_path}...")
    model.load_weights(str(weights_path))

    # Test splits to benchmark
    splits_to_eval = [
        ("Seen Fonts (Moderate)", "test_seen_moderate"),
        ("Unseen Fonts (Font Generalization)", "test_unseen_fonts_moderate"),
        ("Tiers: Clean (Severity 0.0)", "test_tiers_clean"),
        ("Tiers: Moderate (Severity 0.25-0.5)", "test_tiers_moderate"),
        ("Tiers: Hard (Severity 0.7-1.0)", "test_tiers_hard")
    ]

    results = {}
    lookalike_groups = config.get("lookalike_groups", [])

    print("\n" + "=" * 95)
    print(f"  BENCHMARK EVALUATION RESULTS ({weights_path.name})")
    print("=" * 95)
    print(f"{'Split / Condition':<35} | {'Lines':<6} | {'Strict CER':<11} | {'Norm CER':<10} | {'WER':<8} | {'Seq Acc':<8}")
    print("-" * 95)

    for display_name, split_key in splits_to_eval:
        res = evaluate_split(model, dataset_builder, split_key, args.batch_size, lookalike_groups)
        results[split_key] = res
        if "error" in res or res.get("total_lines", 0) == 0:
            print(f"{display_name:<35} | {'N/A':<6} | {'(Not found or empty split)':<40}")
        else:
            print(
                f"{display_name:<35} | "
                f"{res['total_lines']:<6} | "
                f"{res['strict_cer']*100:>8.2f}%  | "
                f"{res['normalized_cer']*100:>7.2f}%  | "
                f"{res['wer']*100:>6.2f}% | "
                f"{res['sequence_accuracy']:>6.2f}%"
            )

    print("=" * 95 + "\n")

    # Save benchmark file
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
