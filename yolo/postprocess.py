"""
postprocess.py - Post-processing for YOLO11 document OCR.
Reconstructs reading-order text lines from 2D character bounding boxes:
  1. Filters character detections by confidence and NMS.
  2. Groups detected boxes into horizontal text lines based on vertical overlap.
  3. Sorts character boxes within each line left-to-right.
  4. Inserts spaces from horizontal gaps (since space is not a detection class).
  5. Computes Strict CER, Normalized CER (look-alikes), and WER against ground truth.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import editdistance


def build_charset_mapping(charset_cfg: Dict[str, Any]) -> Tuple[List[str], Dict[int, str]]:
    """Constructs the 73-class character list and ID-to-char mapping from config."""
    chars: List[str] = []
    chars.extend(list(charset_cfg.get("lowercase", "abcdefghijklmnopqrstuvwxyz")))
    chars.extend(list(charset_cfg.get("uppercase", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
    chars.extend(list(charset_cfg.get("digits", "0123456789")))
    chars.extend(list(charset_cfg.get("punctuation", [".", ",", ";", ":", "!", "?", "'", "\"", "-", "(", ")"])))
    
    id_to_char = {i: c for i, c in enumerate(chars)}
    return chars, id_to_char


def group_boxes_into_lines(
    boxes: List[Dict[str, Any]],
    line_grouping_tol: float = 0.6
) -> List[List[Dict[str, Any]]]:
    """
    Groups 2D character boxes into reading-order lines.
    Each box is a dict: {'x0': float, 'y0': float, 'x1': float, 'y1': float,
                         'cx': float, 'cy': float, 'w': float, 'h': float,
                         'cls': int, 'conf': float, 'char': str}
    """
    if not boxes:
        return []

    # Sort primarily by vertical center
    sorted_by_y = sorted(boxes, key=lambda b: (b["cy"], b["cx"]))
    median_h = float(np.median([b["h"] for b in sorted_by_y])) if sorted_by_y else 18.0
    tol_y = max(4.0, median_h * line_grouping_tol)

    lines: List[List[Dict[str, Any]]] = []
    line_y_centers: List[float] = []

    for box in sorted_by_y:
        matched_line_idx = -1
        min_dist = float("inf")

        # Find best matching line by vertical proximity
        for l_idx, avg_y in enumerate(line_y_centers):
            dist = abs(box["cy"] - avg_y)
            if dist <= tol_y and dist < min_dist:
                min_dist = dist
                matched_line_idx = l_idx

        if matched_line_idx >= 0:
            lines[matched_line_idx].append(box)
            # Update running average vertical center of this line
            line_y_centers[matched_line_idx] = float(np.mean([b["cy"] for b in lines[matched_line_idx]]))
        else:
            lines.append([box])
            line_y_centers.append(box["cy"])

    # Sort lines from top to bottom
    indexed_lines = list(zip(lines, line_y_centers))
    indexed_lines.sort(key=lambda item: item[1])

    # Within each line, sort characters left-to-right
    sorted_lines: List[List[Dict[str, Any]]] = []
    for line, _ in indexed_lines:
        line.sort(key=lambda b: b["cx"])
        sorted_lines.append(line)

    return sorted_lines


def reconstruct_line_text(
    line_boxes: List[Dict[str, Any]],
    space_gap_ratio: float = 0.35
) -> str:
    """
    Reconstructs string text from sorted character boxes by inserting spaces for gaps.
    """
    if not line_boxes:
        return ""

    median_w = float(np.median([b["w"] for b in line_boxes])) if line_boxes else 12.0
    space_threshold = max(3.0, median_w * space_gap_ratio)

    chars: List[str] = [line_boxes[0]["char"]]

    for i in range(1, len(line_boxes)):
        prev_box = line_boxes[i - 1]
        curr_box = line_boxes[i]

        # Horizontal gap between the right edge of previous char and left edge of current char
        gap = curr_box["x0"] - prev_box["x1"]

        # Insert space if horizontal gap exceeds threshold
        if gap >= space_threshold:
            chars.append(" ")

        chars.append(curr_box["char"])

    return "".join(chars)


def reconstruct_page_text(
    boxes: List[Dict[str, Any]],
    line_grouping_tol: float = 0.6,
    space_gap_ratio: float = 0.35
) -> List[str]:
    """
    Takes detected character boxes on a page and returns a list of reconstructed text lines.
    """
    lines = group_boxes_into_lines(boxes, line_grouping_tol=line_grouping_tol)
    return [reconstruct_line_text(line, space_gap_ratio=space_gap_ratio) for line in lines]


def normalize_lookalikes(text: str, lookalike_groups: List[List[str]]) -> str:
    """Canonicalizes ambiguous glyphs (e.g. I/l/1 -> I, O/0/o -> O)."""
    norm_text = list(text)
    mapping = {}
    for group in lookalike_groups:
        if group:
            canonical = group[0]
            for variant in group[1:]:
                mapping[variant] = canonical

    for idx, ch in enumerate(norm_text):
        if ch in mapping:
            norm_text[idx] = mapping[ch]
    return "".join(norm_text)


def compute_transcription_metrics(
    pred_lines: List[str],
    gt_lines: List[str],
    lookalike_groups: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """
    Computes Strict CER, Normalized CER, WER, and Sequence Accuracy between
    predicted lines and ground-truth lines.
    """
    lookalikes = lookalike_groups or []
    n_items = max(len(pred_lines), len(gt_lines))

    # Pad shorter list with empty strings if line counts differ
    p_padded = pred_lines + [""] * (n_items - len(pred_lines))
    g_padded = gt_lines + [""] * (n_items - len(gt_lines))

    total_chars = 0
    total_strict_dist = 0
    total_norm_dist = 0
    total_words = 0
    total_word_dist = 0
    exact_matches = 0

    for p, g in zip(p_padded, g_padded):
        total_chars += len(g)
        strict_d = editdistance.eval(p, g)
        total_strict_dist += strict_d

        p_norm = normalize_lookalikes(p, lookalikes)
        g_norm = normalize_lookalikes(g, lookalikes)
        norm_d = editdistance.eval(p_norm, g_norm)
        total_norm_dist += norm_d

        # Word level
        p_words = p.split()
        g_words = g.split()
        total_words += len(g_words)
        total_word_dist += editdistance.eval(p_words, g_words)

        if p == g:
            exact_matches += 1

    strict_cer = (total_strict_dist / max(1, total_chars)) if total_chars > 0 else 0.0
    norm_cer = (total_norm_dist / max(1, total_chars)) if total_chars > 0 else 0.0
    wer = (total_word_dist / max(1, total_words)) if total_words > 0 else 0.0
    seq_acc = (exact_matches / max(1, len(gt_lines))) * 100 if gt_lines else 0.0

    return {
        "total_lines": len(gt_lines),
        "total_characters": total_chars,
        "total_words": total_words,
        "exact_matches": exact_matches,
        "strict_cer": float(strict_cer),
        "normalized_cer": float(norm_cer),
        "wer": float(wer),
        "sequence_accuracy": float(seq_acc)
    }
