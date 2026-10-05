"""
metrics.py - Evaluation metrics for CRNN OCR pipeline.
Implements:
  - Levenshtein edit distance (character & word level)
  - Strict Character Error Rate (CER)
  - Normalized Character Error Rate (Normalized CER with lookalike groups)
  - Word Error Rate (WER)
  - Sequence Accuracy (Exact Match %)
  - Per-character error breakdown and top confusion analysis
"""

from typing import List, Tuple, Dict, Any, Optional
import collections


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes standard Levenshtein edit distance between two strings or lists.
    Uses O(min(M, N)) space dynamic programming.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def normalize_lookalikes(text: str, lookalike_groups: List[List[str]]) -> str:
    """
    Normalizes look-alike characters (e.g., I/l/1, O/0/o, c/C, s/S, etc.)
    to the first character of each group for fair benchmarking as defined in
    synth_pages README sections 5, 8 & 9.
    """
    mapping = {}
    for group in lookalike_groups:
        if not group:
            continue
        canonical = group[0]
        for char in group:
            mapping[char] = canonical
    
    return "".join(mapping.get(c, c) for c in text)


def compute_cer(
    predictions: List[str],
    ground_truths: List[str],
    lookalike_groups: Optional[List[List[str]]] = None
) -> float:
    """
    Computes Character Error Rate (CER).
    If lookalike_groups is provided, normalizes characters before calculating edit distance.
    CER = sum(edit_distance(gt_i, pred_i)) / sum(len(gt_i))
    """
    if not ground_truths or len(ground_truths) != len(predictions):
        return 0.0

    total_dist = 0
    total_len = 0

    for pred, gt in zip(predictions, ground_truths):
        p = pred
        g = gt
        if lookalike_groups:
            p = normalize_lookalikes(p, lookalike_groups)
            g = normalize_lookalikes(g, lookalike_groups)
        
        dist = levenshtein_distance(g, p)
        total_dist += dist
        total_len += max(len(g), 1)

    return total_dist / max(total_len, 1)


def compute_wer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Computes Word Error Rate (WER) based on space-separated tokens.
    WER = sum(word_edit_distance(gt_words, pred_words)) / sum(len(gt_words))
    """
    if not ground_truths or len(ground_truths) != len(predictions):
        return 0.0

    total_dist = 0
    total_words = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.strip().split()
        gt_words = gt.strip().split()
        
        dist = levenshtein_distance(gt_words, pred_words)
        total_dist += dist
        total_words += max(len(gt_words), 1)

    return total_dist / max(total_words, 1)


def compute_sequence_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Computes exact sequence accuracy (percentage of perfectly transcribed lines).
    """
    if not ground_truths or len(ground_truths) != len(predictions):
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return (correct / len(ground_truths)) * 100.0


def analyze_errors(
    predictions: List[str],
    ground_truths: List[str],
    top_k_confusions: int = 10
) -> Dict[str, Any]:
    """
    Performs detailed error characterization:
    - Counts insertions, deletions, substitutions
    - Identifies top confused character pairs
    """
    substitutions = collections.Counter()
    deletions = collections.Counter()
    insertions = collections.Counter()
    
    total_chars = 0
    exact_matches = 0

    for pred, gt in zip(predictions, ground_truths):
        total_chars += len(gt)
        if pred == gt:
            exact_matches += 1
            continue

        # Use Needleman-Wunsch or simple alignment for character confusion
        # Simple DP backtrack alignment
        m, n = len(gt), len(pred)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if gt[i - 1] == pred[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # deletion from gt
                    dp[i][j - 1] + 1,       # insertion into pred
                    dp[i - 1][j - 1] + cost # substitution or match
                )

        # Backtrack
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if gt[i - 1] == pred[j - 1] else 1):
                if gt[i - 1] != pred[j - 1]:
                    substitutions[(gt[i - 1], pred[j - 1])] += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                deletions[gt[i - 1]] += 1
                i -= 1
            else:
                insertions[pred[j - 1]] += 1
                j -= 1

    top_subs = [
        {"truth": pair[0], "predicted": pair[1], "count": count}
        for pair, count in substitutions.most_common(top_k_confusions)
    ]

    return {
        "total_lines": len(ground_truths),
        "total_characters": total_chars,
        "exact_matches": exact_matches,
        "exact_match_pct": (exact_matches / max(len(ground_truths), 1)) * 100.0,
        "substitution_count": sum(substitutions.values()),
        "deletion_count": sum(deletions.values()),
        "insertion_count": sum(insertions.values()),
        "top_confusions": top_subs
    }


def compute_epoch_metrics(
    predictions: List[str],
    ground_truths: List[str],
    lookalike_groups: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """
    Computes the full metric report for a training or validation epoch.
    """
    strict_cer = compute_cer(predictions, ground_truths, lookalike_groups=None)
    norm_cer = compute_cer(predictions, ground_truths, lookalike_groups=lookalike_groups) if lookalike_groups else strict_cer
    wer = compute_wer(predictions, ground_truths)
    seq_acc = compute_sequence_accuracy(predictions, ground_truths)
    error_analysis = analyze_errors(predictions, ground_truths, top_k_confusions=5)

    return {
        "strict_cer": strict_cer,
        "normalized_cer": norm_cer,
        "cer_reduction_by_norm_pct": max(0.0, ((strict_cer - norm_cer) / max(strict_cer, 1e-6)) * 100.0) if strict_cer > 0 else 0.0,
        "wer": wer,
        "sequence_accuracy": seq_acc,
        "error_analysis": error_analysis
    }
