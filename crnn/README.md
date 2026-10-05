# CRNN OCR Pipeline (TensorFlow + CTC)

Production-grade Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss for line-level OCR, designed to ingest and train directly on the outputs of `synth_pages` / `synth_data`.

---

## 1. Key Architectural & Implementation Features

| Component | Design Choice | Reason |
|---|---|---|
| **CNN Backbone** | 5-stage VGG-style with asymmetric pooling | Reduces height $48 \to 1$ while preserving horizontal sequence length ($W \to W/4$) |
| **Sequence Model** | 2-layer Stacked Bidirectional LSTM ($2 \times 256$ units) | Models bidirectional character contextual dependencies |
| **CTC Loss** | `tf.nn.ctc_loss(..., logits_time_major=False)` | Solves alignment-free sequence labeling |
| **Gradient Clipping** | `clipnorm=5.0` | Prevents gradient explosion in recurrent cells during CTC alignment |
| **Alphabet** | 73 classes + space (`' '`) + CTC blank = 75 classes | Handles alphanumeric text, punctuation, and multi-word sentences |
| **TSV Parsing** | `csv.QUOTE_NONE`, `keep_default_na=False` | Prevents corruption on double quotes (`"`) and words like `"NA"` or `"null"` |
| **Aspect-Ratio Preserve** | Height fixed at 48, right-padding with white (255) | Preserves font glyph aspect ratios and prevents squishing |
| **No Mirror Flips** | Horizontal flip strictly disabled | Mirroring destroys glyph identities (e.g., `d` vs `b`) |
| **Overfitting Defense** | `EarlyStopping(monitor='val_cer')`, `ModelCheckpoint`, `ReduceLROnPlateau` | Halts training when validation Character Error Rate stops improving |
| **Fair Metrics** | Strict CER, Normalized CER (lookalikes), WER, Sequence Accuracy | Accounts for font ambiguities (`I/l/1`, `O/0/o`, `c/C`, etc.) |

---

## 2. Directory Layout & synth_data Integration

The configuration in `training.yaml` directly references the synthetic dataset outputs without copying:

```
workspace/
├── synth_pages/
│   └── output/
│       ├── crnn/
│       │   ├── labels.tsv
│       │   ├── train/
│       │   ├── val/
│       │   ├── test_seen_moderate/
│       │   ├── test_unseen_fonts_moderate/
│       │   ├── test_tiers_clean/
│       │   ├── test_tiers_moderate/
│       │   └── test_tiers_hard/
│       └── metadata.jsonl
└── crnn/
    ├── training.yaml            # Paths point to ../synth_pages/output/crnn
    ├── dataset.py               # Robust TSV parsing & tf.data pipeline
    ├── model.py                 # CRNN architecture + CTC loss
    ├── metrics.py               # Levenshtein distance, Strict & Normalized CER, WER
    ├── callbacks.py             # Per-epoch report callback, EarlyStopping, Checkpoint
    ├── train.py                 # Main training CLI
    ├── evaluate.py              # Test split & tier evaluation CLI
    ├── infer.py                 # Single line & batch inference
    └── requirements.txt
```

---

## 3. Quick Start

### Step 1: Install dependencies
```bash
pip install -r crnn/requirements.txt
```

### Step 2: Test pipeline with dry-run (2 batches)
```bash
python -m crnn.train --config crnn/training.yaml --dry_run
```

### Step 3: Run Full Training
```bash
python -m crnn.train --config crnn/training.yaml --epochs 60 --batch_size 32
```

Outputs generated during training:
- `reports/epoch_001.json`, `epoch_002.json`, ...
- `reports/training_summary.md` (live updated Markdown table)
- `reports/training_report.json`
- `checkpoints/crnn_best_model.weights.h5`

### Step 4: Benchmark on All Splits and Degradation Tiers
```bash
python -m crnn.evaluate \
  --config crnn/training.yaml \
  --weights crnn/checkpoints/crnn_best_model.weights.h5
```

### Step 5: Run Inference on a Line Crop
```bash
python -m crnn.infer \
  --weights crnn/checkpoints/crnn_best_model.weights.h5 \
  --image path/to/sample_line.png \
  --method beam_search
```

---

## 4. Performance & Accuracy Reporting

At every epoch, `EpochAccuracyReportCallback` generates:
1. **Console Table**:
   - CTC training & validation loss
   - Strict CER vs Normalized CER (with look-alike character grouping)
   - Word Error Rate (WER)
   - Sequence Accuracy (% exact matches)
   - Side-by-side ground truth vs decoded predictions
2. **JSON & Markdown Artifacts**:
   - `reports/training_summary.md`
   - `reports/epoch_{N:03d}.json` containing full error confusion analysis.
