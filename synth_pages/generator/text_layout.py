"""Paragraph sampling and line wrapping.

Content is random (uniform letters, no natural words). Structure (word, sentence
and paragraph lengths, punctuation rates) comes from English statistics: the
[FALLBACK] numbers in config.yaml, overridden by assets/english_stats.json when
that file exists.
"""
import json
import math

import numpy as np

_STAT_KEYS = (
    "word_length_probs", "sentence_length_words", "sentences_per_paragraph",
    "comma_prob_per_word", "terminal_punct_probs", "other_punct_prob_per_word",
)


def load_text_params(cfg):
    """Return the text section of the config, overridden by measured stats if present."""
    tp = dict(cfg["text"])
    stats_path = cfg["_root"] / cfg["paths"]["stats_file"]
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        for k in _STAT_KEYS:
            if k in stats:
                tp[k] = stats[k]
    return tp


def _norm(p):
    p = np.asarray(p, dtype=float)
    return p / p.sum()


class TextSampler:
    def __init__(self, tp, charset_cfg):
        self.tp = tp
        self.lower = charset_cfg["lowercase"]
        self.upper = charset_cfg["uppercase"]
        self.digits = charset_cfg["digits"]
        self.punct = set(charset_cfg["punctuation"])
        self.wl_p = _norm(tp["word_length_probs"])
        self.wtypes = list(tp["word_type_probs"])
        self.wt_p = _norm(list(tp["word_type_probs"].values()))
        self.cstyles = list(tp["case_style_probs"])
        self.cs_p = _norm(list(tp["case_style_probs"].values()))
        self.term = [k for k in tp["terminal_punct_probs"] if k in self.punct]
        self.term_p = _norm([tp["terminal_punct_probs"][k] for k in self.term]) if self.term else None
        self.other = [c for c in (";", ":", "'", '"', "-", "(", ")") if c in self.punct]

    #  words 
    def _letters(self, rng, n, alphabet):
        return "".join(alphabet[i] for i in rng.integers(0, len(alphabet), n))

    def word(self, rng):
        wtype = self.wtypes[rng.choice(len(self.wtypes), p=self.wt_p)]
        if wtype == "numeric":
            lo, hi = self.tp["numeric_length_range"]
            return self._letters(rng, int(rng.integers(lo, hi + 1)), self.digits)
        n = 1 + int(rng.choice(len(self.wl_p), p=self.wl_p))
        if wtype == "alnum":
            n = max(n, 3)
            chars = []
            for _ in range(n):
                if rng.random() < 0.5:
                    chars.append(self._letters(rng, 1, self.digits))
                else:
                    c = self._letters(rng, 1, self.lower)
                    chars.append(c.upper() if rng.random() < 0.3 else c)
            s = "".join(chars)
            if not any(c.isdigit() for c in s):
                s = self._letters(rng, 1, self.digits) + s[1:]
            return s
        w = self._letters(rng, n, self.lower)
        style = self.cstyles[rng.choice(len(self.cstyles), p=self.cs_p)]
        if style == "capitalized":
            w = w[0].upper() + w[1:]
        elif style == "upper":
            w = w.upper()
        elif style == "mixed":
            w = "".join(c.upper() if rng.random() < 0.5 else c for c in w)
        return w

    # sentences / paragraphs
    def sentence(self, rng):
        sl = self.tp["sentence_length_words"]
        sigma = 0.5
        mu = math.log(sl["mean"]) - sigma ** 2 / 2
        n = int(np.clip(round(rng.lognormal(mu, sigma)), sl["min"], sl["max"]))
        words = [self.word(rng) for _ in range(n)]
        if self.tp.get("sentence_initial_capital", True) and words[0][0].isalpha():
            words[0] = words[0][0].upper() + words[0][1:]

        # occasional special punctuation
        p_other = self.tp["other_punct_prob_per_word"]
        for i, w in enumerate(words):
            if not self.other or rng.random() >= p_other:
                continue
            c = self.other[rng.integers(0, len(self.other))]
            if c in (";", ":"):
                if i < n - 1:
                    words[i] = w + c
            elif c in ("(", ")"):
                if "(" in self.punct and ")" in self.punct:
                    words[i] = "(" + w + ")"
            elif c == '"':
                words[i] = '"' + w + '"'
            elif c == "'":
                if len(w) >= 3:
                    k = int(rng.integers(1, len(w)))
                    words[i] = w[:k] + "'" + w[k:]
            elif c == "-":
                if len(w) >= 4:
                    k = int(rng.integers(2, len(w) - 1))
                    words[i] = w[:k] + "-" + w[k:]

        # commas
        if "," in self.punct:
            for i in range(n - 1):
                if words[i][-1].isalnum() and rng.random() < self.tp["comma_prob_per_word"]:
                    words[i] += ","

        # terminal punctuation
        if self.term:
            words[-1] += self.term[rng.choice(len(self.term), p=self.term_p)]
        return words

    def paragraph(self, rng):
        sp = self.tp["sentences_per_paragraph"]
        ns = int(rng.integers(sp["min"], sp["max"] + 1))
        words = []
        for _ in range(ns):
            words.extend(self.sentence(rng))
        return words

    def heading(self, rng, hcfg):
        n = int(rng.integers(hcfg["words"]["min"], hcfg["words"]["max"] + 1))
        words = []
        for _ in range(n):
            w = self._letters(rng, 1 + int(rng.choice(len(self.wl_p), p=self.wl_p)), self.lower)
            words.append(w[0].upper() + w[1:])
        return words


def sample_page_blocks(rng, sampler, layout_cfg):
    """A page = 1-3 paragraphs, optionally with one short heading placed before a paragraph."""
    pp = layout_cfg["paragraphs_per_page"]
    n_par = int(rng.integers(pp["min"], pp["max"] + 1))
    blocks = [{"type": "paragraph", "words": sampler.paragraph(rng)} for _ in range(n_par)]
    h = layout_cfg["heading"]
    if rng.random() < h["prob"]:
        pos = int(rng.integers(0, n_par))
        blocks.insert(pos, {"type": "heading", "words": sampler.heading(rng, h)})
    return blocks


def wrap_words(words, measure, max_width, space_w, first_indent=0.0):
    """Greedy line wrapping at a pixel width, like a word processor."""
    lines, cur, cur_w = [], [], 0.0
    limit = max_width - first_indent
    for w in words:
        ww = measure(w)
        if not cur:
            cur, cur_w = [w], ww
        elif cur_w + space_w + ww <= limit:
            cur.append(w)
            cur_w += space_w + ww
        else:
            lines.append(cur)
            cur, cur_w = [w], ww
            limit = max_width
    if cur:
        lines.append(cur)
    return lines