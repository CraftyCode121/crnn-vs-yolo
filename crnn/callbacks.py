"""
callbacks.py - Callbacks for CRNN OCR training:
  - EpochAccuracyReportCallback: Evaluates validation batches every epoch,
    computes Strict CER, Normalized CER, WER, and Sequence Accuracy, logs sample
    predictions, and writes structured JSON & Markdown reports.
  - Injects 'val_cer', 'val_norm_cer', 'val_wer', 'val_seq_acc' into Keras logs
    so EarlyStopping and ModelCheckpoint can directly track character error rate.
  - Checkpointing and EarlyStopping setup to prevent overfitting.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from .metrics import compute_epoch_metrics, normalize_lookalikes
from .dataset import CharsetVocab

logger = logging.getLogger("CRNNCallbacks")


class EpochAccuracyReportCallback(Callback):
    """
    Evaluates validation set at the end of every epoch:
    1. Decodes CTC logits into text sequences via greedy decoding
    2. Computes Strict CER, Normalized CER (lookalike groups), WER, Sequence Accuracy
    3. Prints formatted console summary and side-by-side transcription samples
    4. Writes per-epoch JSON report and updates training_summary.md
    5. Injects computed metrics into epoch logs for EarlyStopping & ModelCheckpoint
    """

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        vocab: CharsetVocab,
        reports_dir: str = "./reports",
        lookalike_groups: Optional[List[List[str]]] = None,
        log_sample_predictions: int = 8,
        eval_batch_limit: Optional[int] = None
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.vocab = vocab
        self.reports_dir = Path(reports_dir)
        self.lookalike_groups = lookalike_groups or []
        self.log_sample_predictions = log_sample_predictions
        self.eval_batch_limit = eval_batch_limit

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.best_cer = float("inf")
        self.best_norm_cer = float("inf")
        self.best_seq_acc = 0.0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        epoch_num = epoch + 1

        all_predictions: List[str] = []
        all_ground_truths: List[str] = []
        sample_comparisons: List[Dict[str, Any]] = []

        batch_count = 0
        for batch in self.val_dataset:
            inputs, _ = batch
            images = inputs["image"]
            labels = inputs["label"].numpy()
            input_length = inputs["input_length"]
            label_length = inputs["label_length"].numpy()

            # Forward pass through model to get logits
            logits = self.model(images, training=False)

            # CTC greedy decoding
            decoded_sparse, _ = self.model.decode(logits, input_length, method="greedy")
            decoded_dense = tf.sparse.to_dense(
                decoded_sparse,
                default_value=self.vocab.blank_idx
            ).numpy()

            batch_size = images.shape[0]
            for i in range(batch_size):
                # Decode predicted sequence
                pred_indices = decoded_dense[i].tolist()
                pred_text = self.vocab.indices_to_text(pred_indices, remove_blank=True)

                # Decode ground-truth sequence
                gt_len = int(label_length[i])
                gt_indices = labels[i, :gt_len].tolist()
                gt_text = self.vocab.indices_to_text(gt_indices, remove_blank=True)

                all_predictions.append(pred_text)
                all_ground_truths.append(gt_text)

                if len(sample_comparisons) < self.log_sample_predictions:
                    sample_comparisons.append({
                        "ground_truth": gt_text,
                        "predicted": pred_text,
                        "exact_match": gt_text == pred_text,
                        "gt_norm": normalize_lookalikes(gt_text, self.lookalike_groups),
                        "pred_norm": normalize_lookalikes(pred_text, self.lookalike_groups)
                    })

            batch_count += 1
            if self.eval_batch_limit and batch_count >= self.eval_batch_limit:
                break

        # Compute all metrics
        metrics = compute_epoch_metrics(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            lookalike_groups=self.lookalike_groups
        )

        strict_cer = metrics["strict_cer"]
        norm_cer = metrics["normalized_cer"]
        wer = metrics["wer"]
        seq_acc = metrics["sequence_accuracy"]

        # Track bests
        is_new_best = strict_cer < self.best_cer
        if is_new_best:
            self.best_cer = strict_cer
            self.best_norm_cer = norm_cer
            self.best_seq_acc = seq_acc

        # Inject into logs for standard Keras callbacks (EarlyStopping, Checkpoint)
        logs["val_cer"] = strict_cer
        logs["val_norm_cer"] = norm_cer
        logs["val_wer"] = wer
        logs["val_seq_acc"] = seq_acc

        train_loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))

        # Console Report Output
        print("\n" + "=" * 78)
        print(f"  CRNN EPOCH {epoch_num:03d} PERFORMANCE & ACCURACY REPORT")
        print("=" * 78)
        print(f"  Training CTC Loss : {train_loss:.4f}  |  Val CTC Loss: {val_loss:.4f}")
        print(f"  Strict CER        : {strict_cer * 100:.2f}%  (Best: {self.best_cer * 100:.2f}%)")
        print(f"  Normalized CER    : {norm_cer * 100:.2f}%  (Lookalike reduction: {metrics['cer_reduction_by_norm_pct']:.1f}%)")
        print(f"  Word Error Rate   : {wer * 100:.2f}%")
        print(f"  Sequence Accuracy : {seq_acc:.2f}% ({metrics['error_analysis']['exact_matches']}/{len(all_ground_truths)} exact match)")
        print("-" * 78)
        print("  Sample Validations:")
        for idx, s in enumerate(sample_comparisons[:5]):
            match_sym = "✓" if s["exact_match"] else "✗"
            print(f"   [{match_sym}] GT:   {s['ground_truth'][:60]}")
            print(f"       PRED: {s['predicted'][:60]}")
        print("=" * 78 + "\n")

        # Compile epoch payload
        epoch_record = {
            "epoch": epoch_num,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "strict_cer": strict_cer,
            "normalized_cer": norm_cer,
            "wer": wer,
            "sequence_accuracy": seq_acc,
            "best_strict_cer_so_far": self.best_cer,
            "sample_predictions": sample_comparisons,
            "error_analysis": metrics["error_analysis"]
        }
        self.history.append(epoch_record)

        # Save per-epoch JSON report
        epoch_file = self.reports_dir / f"epoch_{epoch_num:03d}.json"
        with open(epoch_file, "w", encoding="utf-8") as f:
            json.dump(epoch_record, f, indent=2)

        # Save cumulative training report
        cumulative_file = self.reports_dir / "training_report.json"
        with open(cumulative_file, "w", encoding="utf-8") as f:
            json.dump({
                "epochs_completed": epoch_num,
                "best_strict_cer": self.best_cer,
                "best_normalized_cer": self.best_norm_cer,
                "best_sequence_accuracy": self.best_seq_acc,
                "history": self.history
            }, f, indent=2)

        # Update Markdown Summary Table
        self._write_markdown_summary()

        # Generate publication figures if matplotlib is installed
        if epoch_num % 5 == 0 or epoch_num == 1:
            self._generate_figures()

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Generate final publication evaluation figures at conclusion of training."""
        logger.info("Training complete. Generating publication research figures...")
        self._generate_figures()

    def _generate_figures(self):
        try:
            from .plot_reports import (
                plot_loss_curves,
                plot_error_rate_curves,
                plot_combined_research_panel
            )
            fig_dir = self.reports_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_loss_curves(self.history, fig_dir, fmt="png", dpi=300)
            plot_error_rate_curves(self.history, fig_dir, fmt="png", dpi=300)
            plot_combined_research_panel(self.history, fig_dir, fmt="png", dpi=300)
            logger.info(f"Updated research evaluation figures in {fig_dir}")
        except Exception as e:
            logger.debug(f"Could not generate figures automatically: {e}")

        try:
            from .plotly_dashboard import build_plotly_dashboard
            fig = build_plotly_dashboard(self.history)
            plotly_html_path = self.reports_dir / "plotly_dashboard.html"
            fig.write_html(str(plotly_html_path), include_plotlyjs="cdn", full_html=True)
            logger.info(f"Generated standalone Plotly dashboard at {plotly_html_path}")
        except Exception as e:
            logger.debug(f"Could not generate Plotly dashboard: {e}")

    def _write_markdown_summary(self):
        md_file = self.reports_dir / "training_summary.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# CRNN OCR Training Performance & Accuracy Summary\n\n")
            f.write(f"**Best Strict CER:** `{self.best_cer * 100:.2f}%` | ")
            f.write(f"**Best Normalized CER:** `{self.best_norm_cer * 100:.2f}%` | ")
            f.write(f"**Best Sequence Accuracy:** `{self.best_seq_acc:.2f}%`\n\n")
            f.write("| Epoch | Train Loss | Val Loss | Strict CER | Norm CER | WER | Exact Acc | Top Confusions |\n")
            f.write("|:-----:|:----------:|:--------:|:----------:|:--------:|:---:|:---------:|:---------------|\n")
            for h in self.history:
                top_conf = ", ".join([f"{c['truth']}->{c['predicted']}" for c in h["error_analysis"].get("top_confusions", [])[:3]]) or "None"
                f.write(
                    f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | "
                    f"{h['strict_cer'] * 100:.2f}% | {h['normalized_cer'] * 100:.2f}% | "
                    f"{h['wer'] * 100:.2f}% | {h['sequence_accuracy']:.2f}% | {top_conf} |\n"
                )


def build_callbacks(
    config: Dict[str, Any],
    val_dataset: tf.data.Dataset,
    vocab: CharsetVocab
) -> List[Callback]:
    """
    Assembles a robust suite of callbacks:
      1. EpochAccuracyReportCallback (CER, Normalized CER, WER, per-epoch reports)
      2. EarlyStopping (monitors val_cer to prevent overfitting)
      3. ModelCheckpoint (saves best model by val_cer)
      4. ReduceLROnPlateau (adapts learning rate when loss plateaus)
      5. TensorBoard (graph visualization and scalar traces)
    """
    callbacks_list: List[Callback] = []

    reports_dir = config["paths"].get("reports_dir", "./reports")
    checkpoint_dir = Path(config["paths"].get("checkpoint_dir", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(config["paths"].get("logs_dir", "./logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    reporting_cfg = config.get("reporting", {})
    lookalike_groups = config.get("lookalike_groups", [])

    # 1. Custom Epoch Accuracy and Report Callback
    report_cb = EpochAccuracyReportCallback(
        val_dataset=val_dataset,
        vocab=vocab,
        reports_dir=reports_dir,
        lookalike_groups=lookalike_groups,
        log_sample_predictions=reporting_cfg.get("log_sample_predictions", 8),
        eval_batch_limit=reporting_cfg.get("eval_batch_limit", None)
    )
    callbacks_list.append(report_cb)

    tr_cfg = config.get("training", {})

    # 2. Early Stopping Callback
    es_cfg = tr_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", True):
        callbacks_list.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_cer"),
                patience=es_cfg.get("patience", 10),
                mode=es_cfg.get("mode", "min"),
                restore_best_weights=es_cfg.get("restore_best_weights", True),
                verbose=1
            )
        )

    # 3. Model Checkpointing Callback
    cp_cfg = tr_cfg.get("checkpointing", {})
    if cp_cfg.get("enabled", True):
        model_filename = cp_cfg.get("filename", "crnn_best_model.weights.h5")
        filepath = str(checkpoint_dir / model_filename)
        callbacks_list.append(
            ModelCheckpoint(
                filepath=filepath,
                monitor=cp_cfg.get("monitor", "val_cer"),
                mode=cp_cfg.get("mode", "min"),
                save_best_only=cp_cfg.get("save_best_only", True),
                save_weights_only=True, # Weight checkpointing is maximally robust across TF versions
                verbose=1
            )
        )

    # 4. Learning Rate Plateau Scheduler
    lr_cfg = tr_cfg.get("lr_schedule", {})
    if lr_cfg.get("enabled", True):
        callbacks_list.append(
            ReduceLROnPlateau(
                monitor=lr_cfg.get("monitor", "val_loss"),
                factor=lr_cfg.get("factor", 0.5),
                patience=lr_cfg.get("patience", 4),
                min_lr=float(lr_cfg.get("min_lr", 1e-6)),
                mode=lr_cfg.get("mode", "min"),
                verbose=1
            )
        )

    # 5. TensorBoard Callback
    callbacks_list.append(
        TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch"
        )
    )

    return callbacks_list
