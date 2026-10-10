# Evaluating the Generalization Limits of CRNN-CTC Text Recognition on Synthetic Documents: Font Shift, Degradation Robustness, and a Comparison with Character Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-blueviolet.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official repository for the empirical research study **"Evaluating the Generalization Limits of CRNN-CTC Text Recognition on Synthetic Documents: Font Shift, Degradation Robustness, and a Comparison with Character Detection"** by **Hassan Rasheed**.

This repository contains the complete experimental benchmark suite, comprising **4,900 unique synthetic page layouts** (~47,533 unique underlying line instances) yielding **55,130 evaluated line instances** across training, validation, and five controlled test conditions across 73 typographic classes plus whitespace.

---

## Table of Contents

- [1. Key Verified Empirical Findings](#1-key-verified-empirical-findings)
- [2. Architectural Paradigms: CRNN Transduction vs. YOLO11 Detection](#2-architectural-paradigms-crnn-transduction-vs-yolo11-detection)
- [3. Dataset Generation & Controlled Splits (`synth_pages`)](#3-dataset-generation--controlled-splits-synth_pages)
- [4. Model Architecture & Training Implementation](#4-model-architecture--training-implementation)
- [5. Empirical Benchmark Results](#5-empirical-benchmark-results)
  - [5.1 Main Evaluation Across Test Partitions (CRNN)](#51-main-evaluation-across-test-partitions-crnn)
  - [5.2 Font Shift Evaluation (Seen vs. Unseen Fonts)](#52-font-shift-evaluation-seen-vs-unseen-fonts)
  - [5.3 Degradation Robustness Across Controlled Tiers](#53-degradation-robustness-across-controlled-tiers)
  - [5.4 Error Decomposition & Lookalike Normalization](#54-error-decomposition--lookalike-normalization)
  - [5.5 2D Detection (YOLO11) vs. 1D Sequence Transduction (CRNN)](#55-2d-detection-yolo11-vs-1d-sequence-transduction-crnn)
- [6. Training Dynamics & The Epoch 1 Output Collapse](#6-training-dynamics--the-epoch-1-output-collapse)
- [7. Directory Structure & Organization](#7-directory-structure--organization)
- [8. Quick Start & Reproducibility](#8-quick-start--reproducibility)
- [9. Threats to Validity & Limitations](#9-threats-to-validity--limitations)
- [10. Citation](#10-citation)

---

## 1. Key Verified Empirical Findings

1. **Initial Output Collapse in Epoch 1**: Training undergoes an observed initial output collapse in Epoch 1 where all predictions emit empty strings (`""`), yielding **100% Strict CER, 0.0% sequence accuracy, and 289,413 deletions** (training loss: `57.33`, validation loss: `630.35`). Optimization achieves gradient breakout in Epoch 2 (**4.16% Strict CER, 54.58% sequence accuracy, 835 deletions**), converging to **1.79% validation CER, 1.53% Normalized CER, 7.45% WER, and 68.80% exact line accuracy** at Epoch 12.
2. **Font Generalization Penalty (12.87× CER Ratio)**: In-distribution recognition on seen fonts under moderate degradation achieves **0.31% Strict CER and 89.91% exact line accuracy**. Evaluating on held-out font families increases Strict CER to **3.99%** (a **12.87× error ratio**, +3.67 percentage points), accompanied by a **24.2-fold explosion in character deletions** (from 86 to 2,082).
3. **Homoglyph Disambiguation Limits**: Lookalike character normalization resolves **36.54% of errors on seen fonts** (reducing CER from 0.313% to 0.198%, dominated by optical homoglyphs $I \to l$, $i \to l$, $O \to 0$). However, on unseen fonts, normalization resolves only **5.04% of errors** (from 3.985% to 3.785%), as errors shift to structural glyph misclassifications ($v \to u$, $0 \to o$, $b \to l$) and kerning space insertions ($\text{space} \to l$).
4. **Degradation vs. Font Shift**: Photometric degradation from Clean ($s=0.0$) to Hard ($s=0.70\text{--}1.00$) increases CER from **0.27% to 0.63%** (a **2.33× ratio**, +0.35 percentage points). Error rates under severe synthetic image noise remain substantially lower than those caused by held-out font shifts (0.63% vs. 3.99%).
5. **Localization mAP vs. Text Transcription CER (YOLO11)**: Under the evaluated implementation, high character-detection accuracy (**mAP50 = 94.56%, mAP50–95 = 79.02%**) did **not** translate into accurate reconstructed text transcription (**42.86% CER, WER > 109%**). Slight threshold misalignments in geometric gap-based whitespace inference corrupt word tokenization, compounding Levenshtein distance.

---

## 2. Architectural Paradigms: CRNN Transduction vs. YOLO11 Detection

| Dimension | CRNN Pipeline (`crnn/`) | YOLO11 Pipeline (`yolo/`) |
|---|---|---|
| **Formulation** | 1D sequence transduction with CTC loss | 2D spatial character object detection |
| **Input Representation** | Inverted grayscale text line crops ($H=48\text{ px}$, $W=800\text{ px}$) | Full document page images ($1024\text{ px}$ width, $160\text{--}1024\text{ px}$ height) |
| **Feature Extraction** | 5-stage VGG CNN with asymmetric pooling ($W/4=200$ feature slices) | Ultralytics YOLO11 CSPDarknet + C3k2 + PANet backbone |
| **Sequence Modeling** | 2-layer Stacked BiLSTM ($2 \times 256$ units, 512 total hidden units) | Geometric line clustering + horizontal centroid sorting |
| **Whitespace Handling** | Explicitly trained as vocabulary token (Class ID `73`) | Inferred post-hoc via inter-box geometric gap thresholding |
| **Output Classes** | **75 classes**: 73 alphanumeric/punctuation + space (`73`) + CTC blank (`74`) | **73 classes**: non-space characters only (IDs `0–72`) |
| **Seen Mod. Performance** | **0.31% Strict CER**, **89.91% Seq Acc**, **1.71% WER** | **94.56% mAP50**, **79.02% mAP50-95**, **42.86% CER** |
| **Primary Failure Modes** | Alignment deletion surges under unseen typefaces, CTC collapse | Spacing/gap inference misalignments, punctuation misses |

---

## 3. Dataset Generation & Controlled Splits (`synth_pages`)

Synthetic pages are rendered at $1024\text{ px}$ width with variable heights ($160\text{ to }1024\text{ px}$). Text lines consist of random Latin word sequences uniformly sampled across a **73-character non-space vocabulary**:
- **Lowercase (26)**: `a-z` (IDs `0–25`)
- **Uppercase (26)**: `A-Z` (IDs `26–51`)
- **Digits (10)**: `0-9` (IDs `52–61`)
- **Punctuation (11)**: `. , ; : ! ? ' " - ( )` (IDs `62–72`)

### Dataset Partition Summary (55,130 Total Evaluated Lines)

| Partition | Pages | Lines | Evaluation Role & Content Status |
|---|---|---|---|
| `train` | 3,500 | ~30,000 | Model parameter optimization (unique page layouts; $\ge 40$ font families) |
| `val` | 500 | 6,334 | Checkpointing & early stopping ($\ge 8$ held-out font families) |
| `seen_mod` | 300 | 3,708 | In-distribution baseline under moderate noise (seen fonts) |
| `unseen_mod` | 300 | 3,691 | Font generalization test under moderate noise ($\ge 10$ held-out font families) |
| `tiers_clean` | 300 | 3,800 | Clean control ($s=0.0$) across shared layouts |
| `tiers_mod` | 300 | 3,800 | Moderate control ($s=0.25\text{--}0.50$) across identical shared layouts |
| `tiers_hard` | 300 | 3,797 | Hard noise test ($s=0.70\text{--}1.00$) across identical shared layouts |

*Note: The three `test_tiers_*` partitions reuse the exact same 300 page layouts and 3,800 underlying line instances, isolating photometric degradation from layout variation.*

---

## 4. Model Architecture & Training Implementation

### CRNN Architecture
1. **Input Processing**: Grayscale $[0, 255]$ inverted and normalized to $[0, 1]$, zero-padded with fixed height $H=48$ and max width $W=800$, shape $(B, 48, 800, 1)$.
2. **CNN Feature Extractor**:
   - Stages 1–2: `Conv2D` (64, 128 filters) with standard $2 \times 2$ max-pooling, reducing dimensions to $(12, 200)$.
   - Stages 3–4: `Conv2D` (256, 256 filters) with asymmetric pooling $(2, 1)$, preserving horizontal width at $W/4=200$ while collapsing height.
   - Stage 5: Valid $3 \times 1$ convolution collapsing vertical height to $1$, yielding shape $(B, 1, 200, 512)$.
3. **Map-to-Sequence Squeeze**: Tensor is squeezed to $(B, 200, 512)$ and projected through a Dense layer ($128$ units, ReLU, $0.25$ dropout).
4. **Stacked BiLSTM**: 2-layer stacked Bidirectional LSTM with $256$ units per direction ($512$ concatenated hidden units).
5. **CTC Output Layer**: Linear projection to $75$ classes (73 alphanumeric/punctuation glyphs, whitespace index `73`, CTC blank token index `74`).

### Training Hyperparameters
- **Loss**: Connectionist Temporal Classification loss ($\mathcal{L}_{\text{CTC}} = -\ln P(Y|X)$).
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}$), initial learning rate $1 \times 10^{-3}$.
- **Gradient Clipping**: Norm $5.0$.
- **Batch Size**: 32.
- **Duration**: 12 epochs.

---

## 5. Empirical Benchmark Results

### 5.1 Main Evaluation Across Test Partitions (CRNN)

Evaluated across all controlled test conditions on 55,130 line instances:

| Partition | Strict CER | Norm CER | WER | Exact Seq. Acc. |
|---|---|---|---|---|
| **Seen Mod.** (`test_seen_moderate`) | **0.31%** | **0.20%** | **1.71%** | **89.91%** |
| **Unseen Mod.** (`test_unseen_fonts_moderate`) | **3.99%** | **3.78%** | **7.65%** | **81.60%** |
| **Tiers: Clean** ($s = 0.0$) | **0.27%** | **0.15%** | **1.43%** | **90.63%** |
| **Tiers: Mod.** ($s = 0.25\text{--}0.50$) | **0.25%** | **0.12%** | **1.34%** | **91.21%** |
| **Tiers: Hard** ($s = 0.70\text{--}1.00$) | **0.63%** | **0.47%** | **3.11%** | **83.72%** |

---

### 5.2 Font Shift Evaluation (Seen vs. Unseen Fonts)

Comparing `test_seen_moderate` against `test_unseen_fonts_moderate` under identical moderate degradation reveals:
- **Strict CER**: Escalates from $0.31\%$ ($0.00313$) to $3.99\%$ ($0.03985$), a **$12.87\times$ error ratio** ($+3.67$ percentage points).
- **Normalized CER**: Escalates from $0.20\%$ ($0.00198$) to $3.78\%$ ($0.03785$), a **$19.12\times$ error ratio** ($+3.59$ percentage points).
- **Word Error Rate (WER)**: Rises from $1.71\%$ to $7.65\%$ (a **$4.47\times$ ratio**, $+5.94$ percentage points).
- **Sequence Accuracy**: Drops by $8.31$ percentage points (from $89.91\%$ to $81.60\%$), representing a relative decrease of $9.24\%$.

---

### 5.3 Degradation Robustness Across Controlled Tiers

Across the identical 300 page layouts in `test_tiers_*`:
- Moving from **Clean ($s=0.0$) to Hard ($s=0.70\text{--}1.00$)** increases Strict CER from **$0.27\%$ to $0.63\%$**, a **$2.33\times$ ratio** ($+0.35$ percentage points).
- Exact line sequence accuracy drops from **$90.63\%$ to $83.72\%$** ($-6.91$ percentage points).
- **Key Finding**: In this benchmark, the error rate under Hard degradation on seen fonts ($0.63\%$ CER) remains substantially lower than that on unseen fonts under moderate degradation ($3.99\%$ CER).

---

### 5.4 Error Decomposition & Lookalike Normalization

#### Levenshtein Error Operations Breakdown:

| Partition | Substitutions | Deletions | Insertions | Total Errors |
|---|---|---|---|---|
| **Seen Mod.** | 386 ($68.8\%$) | 86 ($15.3\%$) | 89 ($15.9\%$) | 561 |
| **Unseen Mod.** | 4,808 ($65.1\%$) | **2,082 ($28.2\%$)** | 494 ($6.7\%$) | 7,384 |
| **Tiers Clean** | 367 ($73.7\%$) | 70 ($14.1\%$) | 61 ($12.2\%$) | 498 |
| **Tiers Hard** | 818 ($71.3\%$) | 227 ($19.8\%$) | 103 ($9.0\%$) | 1,148 |

#### Lookalike Normalization Protocol:
Characters are canonicalized into eight equivalence groups:
`[I, l, 1] → l`, `[O, 0, o] → 0`, `[c, C] → c`, `[s, S] → s`, `[v, V] → v`, `[w, W] → w`, `[x, X] → x`, `[z, Z] → z`.

- **On Seen Fonts**: Normalization reduces CER by **$36.54\%$** (from $0.313\%$ to $0.198\%$). Errors are dominated by vertical homoglyphs:
  - $I \to l$ (99 errors), $i \to l$ (64 errors), $O \to 0$ (37 errors).
- **On Unseen Fonts**: Normalization reduces CER by only **$5.04\%$** (from $3.985\%$ to $3.785\%$). Errors shift to:
  - Structural glyph misclassifications: $v \to u$ (159 errors), $0 \to o$ (70 errors), $b \to l$ (62 errors).
  - Kerning space insertions: $\text{space} \to l$ (91 errors).
  - **$24.2\times$ Deletion Surge**: Character omissions explode from 86 to 2,082. Unfamiliar glyph geometries fail to trigger confident emissions during CTC decoding.

---

### 5.5 2D Detection (YOLO11) vs. 1D Sequence Transduction (CRNN)

Direct comparison between full-page YOLO11 character detection and line-level CRNN sequence recognition on identical document layouts:

| Partition | YOLO11 mAP50 | YOLO11 mAP50–95 | YOLO11 CER | CRNN CER |
|---|---|---|---|---|
| **Seen Mod.** | 94.56% | 79.02% | **42.86%** | **0.31%** |
| **Unseen Mod.** | 89.98% | 74.15% | **46.18%** | **3.99%** |
| **Tiers Clean** | 94.32% | 78.84% | **43.79%** | **0.27%** |
| **Tiers Hard** | 91.99% | 76.51% | **44.27%** | **0.63%** |

*Analysis*: While YOLO11 achieves strong spatial localization accuracy ($\text{mAP50} = 94.56\%$), its reconstructed text line transcription CER exceeds $42\%$ ($\text{WER} > 109\%$). Object detection mAP evaluates isolated bounding-box overlap, whereas CER evaluates sequential string fidelity. Because whitespace lacks physical bounding boxes, geometric gap thresholding misalignments corrupt tokenization.

---

## 6. Training Dynamics & The Epoch 1 Output Collapse

Inspection of the 12-epoch training trajectory:

- **Epoch 1 (Initial Output Collapse)**:
  - Training Loss: `57.33` | Validation Loss: `630.35`
  - Strict CER: `100.0%` | Sequence Accuracy: `0.0%`
  - Deletion Count: `289,413` (100% of target characters omitted; every prediction emitted as empty string `""`).
  - *Theoretical Hypothesis*: Consistent with CTC optimization theory, near-uniform initial softmax logits make emitting the CTC blank token across all frames a low-entropy local attractor before coherent temporal alignment gradients emerge.
- **Epoch 2 (Gradient Breakout)**:
  - Optimization escapes the blank attractor: Strict CER drops to `4.16%`, exact sequence accuracy rises to `54.58%`, and deletion errors plunge from 289,413 to `835`.
- **Epoch 12 (Convergence)**:
  - Validation Checkpoint on held-out `val/` font families: **Strict CER: 1.79%**, **Normalized CER: 1.53%**, **WER: 7.45%**, **Line Sequence Accuracy: 68.80%** (4,358 of 6,334 lines matched).
  - Training Loss: `0.51` | Validation Loss: `3.75`.

---

## 7. Directory Structure & Organization

```text
crnn-vs-yolo/
├── synth_pages/                 # Synthetic page generator (Section 4)
│   ├── config.yaml              # Typography, layout parameters, degradations
│   ├── build.py                 # Multi-process page renderer & exporter
│   ├── generator/               # Layout, PIL rendering, degrade, export modules
│   ├── tools/
│   │   ├── split_fonts.py       # Strict train (≥40), val (≥8), unseen (≥10) font splitter
│   │   ├── check_dataset.py     # 11 validation sanity checks
│   │   └── visualize_boxes.py   # Visual inspection utility
│   └── output/                  # Paired dataset: yolo/ boxes & crnn/ line crops
│
├── crnn/                        # TensorFlow CRNN OCR module (Section 5)
│   ├── training.yaml            # Hyperparameters & data paths
│   ├── dataset.py               # TSV parser & tf.data pipeline
│   ├── model.py                 # 5-stage VGG + 2-layer BiLSTM + CTC model
│   ├── metrics.py               # Levenshtein distance, Strict/Norm CER, WER
│   ├── callbacks.py             # Validation logging & checkpointing
│   ├── train.py                 # Training orchestrator CLI
│   └── evaluate.py              # Five-condition benchmark evaluation CLI
│
└── yolo/                        # Ultralytics YOLO11 character detector module (Section 6)
    ├── training.yaml            # YOLO fine-tuning configuration
    ├── dataset.py               # Dynamic data.yaml generator
    ├── postprocess.py           # Geometric line grouping & gap space inference
    ├── train.py                 # YOLO11 fine-tuning CLI
    └── evaluate.py              # Dual benchmark CLI (mAP50/mAP50-95 + CER/WER)
```

---

## 8. Quick Start & Reproducibility

### Step 1: Environment Setup & Dataset Generation

```bash
git clone https://github.com/CraftyCode121/crnn-vs-yolo.git
cd crnn-vs-yolo

# Setup synth_pages & download/split fonts
cd synth_pages
pip install -r requirements.txt
python tools/split_fonts.py --download 120

# Run full dataset generation across 4 workers
python build.py --clean --workers 4
python tools/check_dataset.py
cd ..
```

### Step 2: Train & Evaluate CRNN

```bash
pip install -r crnn/requirements.txt

# Train for 12 epochs
python -m crnn.train --config crnn/training.yaml

# Evaluate on all 5 test conditions
python -m crnn.evaluate --config crnn/training.yaml
```

### Step 3: Train & Evaluate YOLO11

```bash
pip install -r yolo/requirements.txt

# Verify dataset & generate data.yaml
python -m yolo.dataset --config yolo/training.yaml --verify

# Train YOLO11
python -m yolo.train --config yolo/training.yaml --model_path yolo11n.pt

# Evaluate dual metrics (mAP + CER/WER)
python -m yolo.evaluate --config yolo/training.yaml --weights yolo/checkpoints/best.pt
```

---

## 9. Threats to Validity & Limitations

1. **Synthetic Domain Shift**: Evaluations use synthetic document images from `synth_pages`. Generalization to physical scanned documents remains unmeasured.
2. **Lack of Language Model Prior**: Text lines are sampled uniformly at random across vocabulary characters without lexical n-gram priors. This isolates visual sequence recognition, but prevents semantic disambiguation.
3. **Confusion Analysis Scope**: Available evaluation logs record the top-5 confusion pairs per split. Hypothesized pairs outside the top-5 threshold could not be evaluated.
4. **Absence of Line-Length Partitioning**: Error metrics are aggregated across line crops without stratifying by sequence length.
5. **Unavailable Internal Activations**: Continuous frame-level softmax activations and hidden recurrent states were not logged during training. Blank-token attractors and alignment failure modes represent plausible hypotheses grounded in CTC optimization theory rather than directly confirmed internal activations.

---

## 10. Citation

If you use this benchmark, code, or experimental data, please cite the research paper:

```bibtex
@article{rasheed2026evaluating,
  title={Evaluating the Generalization Limits of CRNN-CTC Text Recognition on Synthetic Documents: Font Shift, Degradation Robustness, and a Comparison with Character Detection},
  author={Hassan, Rasheed},
  journal={Research Paper: Empirical Evaluation of Document Text Recognition},
  year={2026},
  url={https://github.com/CraftyCode121/crnn-vs-yolo}
}
```
