# synth_pages

A synthetic document-OCR dataset generator. It renders pages of random text with realistic
paragraph structure, degrades them (old paper, stains, skew, blur, noise, JPEG), and writes **two
label formats from the same pages**:

| Format | For | Unit |
|---|---|---|
| **YOLO** | character detectors (YOLOv8 / YOLO11, etc.) | one image per page, one box per character |
| **CRNN** | line recognizers (CNN + RNN + CTC) | one fixed-height image per text line, plus its text |

It was built to compare a CRNN against a YOLO11 character detector on identical content, but the
output can train and evaluate any OCR pipeline.

If you only want to **use** an already generated dataset, read sections 3, 4 and 5 and the
pitfalls in section 9. If you want to **generate** one, read section 6.

---

## 1. Quick start

```bash
pip install -r requirements.txt            # numpy, pillow, opencv-python, scipy, pyyaml

python tools/split_fonts.py --download 120 # get fonts and split them (or: --src /usr/share/fonts)
python build.py --pilot --clean            # small test run (a few minutes)
python tools/visualize_boxes.py --folder train --n 4 --labels
python tools/check_dataset.py              # 11 sanity checks

python build.py --clean --workers 4        # full build (see section 6 for sizes and time)
python tools/check_dataset.py
```

---

## 2. Project layout

```
synth_pages/
├── config.yaml                 all settings (charset, layout, degradations, splits, export)
├── build.py                    entry point: runs the whole pipeline
├── requirements.txt
├── assets/
│   ├── fonts/
│   │   ├── train/              fonts used for training (and the "seen font" test sets)
│   │   ├── val/                fonts used for validation
│   │   ├── unseen_test/        fonts never used in training
│   │   ├── _downloaded/        download cache (not used directly by the generator)
│   │   └── fonts_manifest.json exact font list + split + hashes (for reproduction)
│   └── english_stats.json      OPTIONAL measured English statistics (see 6.3)
├── generator/
│   ├── __init__.py             load_config(), charset_list()
│   ├── text_layout.py          random paragraphs, punctuation, line wrapping
│   ├── render.py               page layout, font handling, per-character rendering
│   ├── degrade.py              geometric + photometric degradations (severity 0..1)
│   └── export.py               YOLO labels, CRNN line crops, data.yaml
├── tools/
│   ├── split_fonts.py          download/collect fonts, split them by family
│   ├── visualize_boxes.py      draw YOLO boxes on sample images
│   └── check_dataset.py        sanity checks on a generated dataset
└── output/                     GENERATED, never edit by hand, not in git
    ├── yolo/
    │   ├── data.yaml
    │   ├── images/<folder>/<page_id>.png
    │   └── labels/<folder>/<page_id>.txt
    ├── crnn/
    │   ├── labels.tsv
    │   └── <folder>/<page_id>_L<NN>.png
    ├── metadata.jsonl
    ├── dataset_info.json
    └── preview/                from visualize_boxes.py
```

---

## 3. What the dataset contains

### 3.1 Splits and folder names

Pages are the unit. A *split* is defined in `config.yaml`. Each split writes into one or more
*folders*, named `<split>` or `<split>_<tier>`.

| Folder (use this name) | Fonts from | Severity | Purpose |
|---|---|---|---|
| `train` | `train/` | random in [0, 1] | training |
| `val` | `val/` | random in [0, 1] | validation, tuning post-processing |
| `test_seen_moderate` | `train/` | moderate | new text, fonts seen in training |
| `test_unseen_fonts_moderate` | `unseen_test/` | moderate | **font generalization** (the only unseen-font test) |
| `test_tiers_clean` | `train/` | clean (0.0) | same pages as the two folders below |
| `test_tiers_moderate` | `train/` | moderate | |
| `test_tiers_hard` | `train/` | hard | |

Default page counts (as shipped in `config.yaml`, change them freely):
train 3500, val 500, test_seen 300, test_unseen_fonts 300, test_tiers 300 (x3 severities).
That is about 5,500 page images and roughly 40,000 to 50,000 CRNN line crops, around **7 GB**.

**`test_tiers_*` shares pages.** Page `test_tiers_000007` exists in all three folders with
identical text and layout and only the degradation differs. This makes severity comparisons
fair. It also means these three folders must never be split apart for training or validation.

### 3.2 Severity

One number in [0, 1] drives every degradation. It is stored per image (`severity` in
`metadata.jsonl` and `labels.tsv`).

| Tier | Severity | What you get |
|---|---|---|
| clean | exactly 0.0 | no warping, no damage (only slight random paper tone and ink darkness) |
| moderate | 0.25 to 0.5 | mild skew, noise, blur, paper texture |
| hard | 0.7 to 1.0 | skew up to about 7 degrees, stains, blur, JPEG artifacts, low resolution, uneven lighting |

Effects at severity 1.0 (all configurable): skew 7 deg, stretch 15%, warp 3 px, perspective 2%,
paper texture, up to 6 stains, ink bleed, resolution drop to 70%, blur sigma 1.0,
noise sigma 15, contrast loss 30%, JPEG quality down to 45.

### 3.3 Content

- **Characters:** 73 classes: `a-z`, `A-Z`, `0-9` and the punctuation `. , ; : ! ? ' " - ( )`.
  The space is part of the text but **not a class** (see section 9).
- **Text is random, not natural language.** Letters are uniform, so there is no language prior.
  Word lengths, sentence lengths, paragraph sizes and punctuation rates follow English statistics
  (fallback values in `config.yaml`, or measured values from `assets/english_stats.json`).
- **Words:** about 92% letters-only, 5% digits, 3% mixed letters and digits. Letter words are 70%
  lowercase, 15% Capitalized, 10% ALL CAPS, 5% mixed case. The first word of each sentence is
  capitalized. Uppercase is about 17% of letters.
- **Layout:** 1 to 3 paragraphs per page, optional short heading, 1 to 15 lines, left-aligned
  (85%) or justified (15%), random indent, margins, column width, font size (10 to 16 pt at
  150 dpi) and line spacing. Lines come from wrapping paragraphs, so the last line of a paragraph
  is naturally short.
- **Canvas:** width is always 1024 px, height varies from 160 to 1024 px. Images are RGB PNG.

---

## 4. Output formats

### 4.1 YOLO (`output/yolo/`)

```
images/<folder>/<page_id>.png        RGB, 1024 px wide, 160..1024 px tall
labels/<folder>/<page_id>.txt        one line per character
data.yaml                            class names and folder paths
```

**Label line** (standard YOLO, normalized to the image size):

```
<class_id> <x_center> <y_center> <width> <height>
```

- One box per **non-space** character, tight to the ink plus 1 px padding. Boxes are derived from
  the warped label map, so they stay correct after skew and warp.
- Lines appear in rendering order (reading order), not sorted by position.
- Tiny classes exist: punctuation boxes can be only 3 to 4 px tall. Median character box height is
  about 18 px.

**Class ids** (`charset` order in `config.yaml`):

| ids | characters |
|---|---|
| 0 to 25 | `a` to `z` |
| 26 to 51 | `A` to `Z` |
| 52 to 61 | `0` to `9` |
| 62 to 72 | `.` `,` `;` `:` `!` `?` `'` `"` `-` `(` `)` |

**`data.yaml`:**

```yaml
path: /absolute/path/to/output/yolo     # ABSOLUTE, written at build time
train: images/train
val: images/val
test:                                   # list of all test folders
- images/test_seen_moderate
- images/test_tiers_clean
...
names: {0: a, 1: b, ..., 72: ')'}
```

### 4.2 CRNN (`output/crnn/`)

```
<folder>/<page_id>_L<NN>.png    one text line, RGB PNG, height exactly 48 px, variable width
labels.tsv                      tab-separated, with a header row
```

Each crop is the rectified line (cut along the line's own direction, so skewed pages do not leak
neighbouring lines), padded by 6 px and resized to height 48 with the aspect ratio preserved. At
tight line spacing, ascenders or descenders of the adjacent line can still appear at the edge.

**`labels.tsv` columns:**

| Column | Meaning |
|---|---|
| `image_path` | path relative to `output/crnn/`, e.g. `train/train_000012_L03.png` |
| `text` | the line's text, single spaces between words |
| `font` | font file used for the page body |
| `split` | **the folder name**, e.g. `test_tiers_hard` (not the base split, see 4.3) |
| `severity` | degradation severity of this image |
| `page_id` | source page, e.g. `train_000012` |
| `line` | 0-based line index within the page |

Texts contain `"` and `'`. **Read the file with quoting disabled** (section 5).

**Alphabet for CTC:** the 73 characters above **plus the space**, plus your blank symbol.

### 4.3 `metadata.jsonl`

One JSON object per output image (one per folder-variant). Useful for breaking results down by
condition.

| Field | Meaning |
|---|---|
| `page_id` | `<split>_<index:06d>`, e.g. `test_tiers_000007` |
| `split` | base split from the config (`test_tiers`) |
| `folder` | output folder (`test_tiers_hard`) |
| `tier` | `random`, `clean`, `moderate` or `hard` |
| `width`, `height` | image size in px |
| `font_body`, `font_style`, `size_pt`, `size_px` | body font file, style, size |
| `line_spacing`, `alignment`, `margin_px`, `column_px`, `indent` | layout |
| `n_lines`, `n_chars`, `n_boxes` | counts (`n_boxes` equals `n_chars` when nothing was lost) |
| `severity` | degradation severity (0 to 1) |
| `skew_deg`, `stretch_x`, `stretch_y`, `warp_px`, `persp`, `fit_scale` | geometric parameters actually used |
| `ink_bleed`, `paper`, `stains`, `lighting`, `contrast_loss`, `low_res`, `blur`, `noise`, `jpeg_q` | photometric parameters, **only present when severity is above 0** (`jpeg_q`: above 0.05) |
| `lines` | the ground-truth text of each line, in reading order |

**Naming trap:** in `metadata.jsonl`, `split` is the base split and `folder` is the output
folder. In `labels.tsv`, the column called `split` holds the **folder**.

**Page-level ground truth:** `"\n".join(lines)`. Heading lines are included as ordinary lines.
Paragraph breaks and headings are not marked.

### 4.4 Reproducibility files

- `output/dataset_info.json`: seed, hash of the config, hash of all font files, image count and
  library versions used for this build.
- `assets/fonts/fonts_manifest.json`: every font file with its split, category, source path,
  pinned Google Fonts commit and SHA-256.

---

## 5. Using the data

### Load the CRNN labels (read this carefully)

```python
import csv, pandas as pd

df = pd.read_csv(
    "output/crnn/labels.tsv",
    sep="\t",
    quoting=csv.QUOTE_NONE,     # texts contain " and '
    keep_default_na=False,      # so a line like "NA" or "null" is not turned into NaN
    dtype={"text": str},
)
train = df[df["split"] == "train"]       # "split" is the FOLDER name
```

### Read YOLO labels

```python
import cv2
from pathlib import Path

root = Path("output/yolo")
img = cv2.imread(str(root / "images/train/train_000000.png"))
H, W = img.shape[:2]
for line in (root / "labels/train/train_000000.txt").read_text().splitlines():
    c, cx, cy, w, h = line.split()
    x0, y0 = (float(cx) - float(w) / 2) * W, (float(cy) - float(h) / 2) * H
    ...
```

### Train YOLO (example with Ultralytics, check flags for your version)

```bash
yolo detect train data=output/yolo/data.yaml model=yolo11s.pt imgsz=1024 fliplr=0.0
```

- **Use `imgsz=1024`** (the native width). Characters are small, and downscaling destroys
  punctuation.
- **Disable horizontal flips** (`fliplr=0.0`). A mirrored glyph is a different shape, so flips
  corrupt the labels. Be equally careful with any other mirror or vertical-flip augmentation.
- If you move or zip the dataset, edit `path:` in `data.yaml` (it is absolute).
- Ultralytics writes `labels.cache` files next to the labels; they are safe to delete.

### Rebuild text from YOLO detections

YOLO gives characters, not text. Your post-processing must group boxes into lines (by vertical
position), sort left to right, and **insert spaces from gaps**, because the space is not a class.
Tune its thresholds on `val` only, never on a test folder.

### Evaluate fairly

- Score every system on the **same folders** with **one** shared scoring script.
- Report **CER** (character error rate) as the main metric, with WER as a secondary one.
- **Line level** (both models get the same line crops, or the lines recovered from boxes) and
  **page level** (full pipeline) answer different questions. Report them separately.
- **Look-alikes:** `I/l/1`, `O/0/o`, `c/C`, `s/S`, `v/V`, `w/W`, `x/X`, `z/Z` are identical or nearly
  identical in some fonts, and the text is random, so there is no context to disambiguate. The
  groups are listed under `lookalike_groups` in `config.yaml`. Report **strict CER** and
  **normalized CER** (each group counted as one class).
- Compare `test_tiers_clean`, `test_tiers_moderate` and `test_tiers_hard` page by page (same
  `page_id`) to measure how each model degrades.
- Only `test_unseen_fonts_moderate` measures font generalization. `test_seen_moderate` and
  `test_tiers_*` use fonts the training split also uses.
- The ground truth is synthetic random text in Latin script. Results do not automatically transfer
  to real documents.

---

## 6. Generating the dataset

### 6.1 Fonts

The generator reads fonts from `assets/fonts/{train,val,unseen_test}/`. The splitter puts all
styles of one family in the same split, so unseen fonts are genuinely unseen, and it spreads font
categories (serif, sans-serif, monospace) across the splits.

```bash
python tools/split_fonts.py --download 120               # from Google Fonts (needs internet)
python tools/split_fonts.py --src /usr/share/fonts ~/.fonts   # from this machine
python tools/split_fonts.py --download 120 --src ~/fonts --clear   # combine, redo the split
python tools/split_fonts.py --from-manifest assets/fonts/fonts_manifest.json   # exact copy of someone else's set
```

Fonts missing any needed glyph are skipped automatically. The config recommends at least 40 files
in `train`, 8 in `val` and 10 in `unseen_test`. Variable fonts render at their default weight.
Check each font's licence if you redistribute images.

### 6.2 Build

```bash
python build.py                       # full build from config.yaml
python build.py --pilot               # about 1% of each split (at least 20 pages)
python build.py --pages 50            # force 50 pages per split
python build.py --splits train val    # only some splits
python build.py --clean               # delete old yolo/crnn/metadata output first
python build.py --workers 8           # parallel processes
```

- **Speed:** about 0.18 s per image on one core. The default config (about 5,500 images) is
  roughly 17 minutes on one core, or about 5 minutes on four.
- **Disk:** about 0.8 MB per page image and 70 KB per line crop (about 1.3 MB per page in total).
  Keep at least 10 GB free for the default config.
- **Run the final build as one command.** `metadata.jsonl`, `labels.tsv` and `data.yaml` are
  written at the end and describe **only that run**. A `--pilot`, `--pages` or `--splits` run
  overwrites them.
- **There is no resume.** If a build stops, run the same command again. Generation is
  deterministic and reproduces the same files.

### 6.3 Configuration (`config.yaml`)

| Section | Controls |
|---|---|
| `seed`, `workers` | reproducibility, parallelism |
| `paths` | font folders, stats file, output folder |
| `charset` | the characters and class order |
| `text` | word types, case styles, word, sentence and paragraph statistics, punctuation rates |
| `layout` | canvas, margins, column width, paragraphs per page, line limits, font size, spacing, alignment, headings |
| `fonts` | allowed extensions, minimum fonts per split, regular/bold/italic mix |
| `lookalike_groups` | groups used for normalized CER |
| `degradation` | severity tiers and the maximum strength of every effect |
| `splits` | pages per split, which fonts, which severity |
| `pilot` | scale for `--pilot` |
| `export` | image format, CRNN line height and padding, labels file name |


**Legibility warning.** Text this small becomes unreadable if you push several degradations up
together. Past versions at `blur_sigma: 2.0` with a 50% resolution drop were unreadable for
9 pt text. After any change to `degradation`, look at a `test_tiers_hard` image at 100% zoom.

### 6.4 How it works

1. `text_layout.py` samples paragraphs (random characters, English-like structure).
2. `render.py` picks a font and layout, wraps lines, and draws every character individually into an
   ink layer and a per-character label map.
3. `degrade.py` warps the ink **and** the label map with the same mapping, then applies photometric
   effects to the ink only.
4. `export.py` derives boxes from the warped label map and writes both formats. Every page is
   rendered once; `test_tiers` degrades that one render three times.

---

## 7. Reproducibility

- Every page has its own random seed derived from `(seed, split, page index)`. Any single page
  can be regenerated exactly, and raising a split's page count later only **adds** pages.
- Same config + same fonts + same library versions reproduces the data **pixel for pixel on the
  same machine** (checked by `check_dataset.py`).
- Other machines can differ by tiny pixel amounts (FreeType and JPEG library versions), even though
  the statistics are the same.
- Fonts are the weak point. Downloads are pinned to a Google Fonts commit and recorded in
  `fonts_manifest.json`; restore them with `--from-manifest`, which verifies SHA-256 hashes.
  Fonts that came from a local folder (`--src`) cannot be downloaded by someone else.
- Keep `config.yaml`, `fonts_manifest.json`, `output/dataset_info.json` and a `pip freeze` next to
  any reported result.

---

## 8. Tools

| Command | Does |
|---|---|
| `python tools/visualize_boxes.py --folder train --n 4 [--labels]` | draws boxes on samples into `output/preview/` |
| `python tools/check_dataset.py` | 11 checks: label format, box counts, class coverage, tier consistency, font leakage, CRNN text vs metadata, crop height, exact regeneration; also prints class balance and box sizes |
| `python tools/split_fonts.py ...` | see 6.1 |

---

## 9. Pitfalls checklist

1. **`labels.tsv` quoting.** Read with `quoting=csv.QUOTE_NONE` and `keep_default_na=False`. The
   default CSV or pandas settings silently corrupt lines containing `"`.
2. **`split` means different things.** `labels.tsv`: the folder name. `metadata.jsonl`: the base
   split (use `folder` there).
3. **No space class in YOLO.** Rebuild spaces from gaps and tune that only on `val`.
4. **Disable flip augmentations** for YOLO training.
5. **`data.yaml` has an absolute `path`.** Fix it after moving the dataset.
6. **Do not train on any `test_*` folder**, and keep the three `test_tiers_*` folders together:
   they contain the same pages.
7. **"Seen font" tests are not font generalization.** Only `test_unseen_fonts_moderate` is.
8. **Look-alike characters** limit accuracy for every model. Report strict and normalized CER.
9. **Image heights vary** (160 to 1024 px); the width is always 1024. CRNN crops have a fixed
   height of 48 px and variable width.
10. **Tiny boxes.** Dots and commas can be 3 to 4 px tall; they are the hardest YOLO classes.
11. **Rare punctuation.** `-`, `!` and `'` appear far less often than letters. Raise
    `other_punct_prob_per_word` in the config if you need more of them.
12. **Partial builds overwrite the bookkeeping files** (section 6.2).
13. **Fonts and output are not in git.** Rebuild fonts with the manifest and the data with
    `build.py`.

---

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ImportError: cannot import name 'charset_list'` | `generator/__init__.py` is still an empty placeholder; replace it with the delivered file. Check with `wc -l generator/*.py`: a 1 or 2 line file is a placeholder. |
| `No usable fonts in ...` | the font folder is empty or every font lacks a needed glyph; run `split_fonts.py` |
| SSL or `403` error while downloading fonts | rate limit or a flaky connection; retry, turn off VPN or proxy, set `GITHUB_TOKEN`, or use `--src` with local fonts |
| `fewer than recommended` warning | too few font files; get more fonts, the models will generalize less otherwise |
| Text unreadable in `test_tiers_hard` | degradations too strong; lower `blur_sigma`, `low_res_scale`, or raise `font_size_pt` |
| Labels look shifted on warped images | run `visualize_boxes.py` and `check_dataset.py`; report with the image and its `metadata.jsonl` line |
| Labels with `"` look wrong in your own loader | see pitfall 1 |
| Out of disk space | lower `splits.*.pages` in the config; the size is roughly 1.3 MB per page |