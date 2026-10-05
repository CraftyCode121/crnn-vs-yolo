"""
dataset.py - Dataset loader and pre-processing pipeline for CRNN OCR model.
Adheres to synth_pages / synth_data specifications:
  - Correct TSV parsing with quoting=csv.QUOTE_NONE, keep_default_na=False
  - Space character inclusion and CTC blank symbol handling
  - Aspect-ratio preserving line resize with right-padding for height=48 crops
  - tf.data pipeline with batching, prefetching, and data augmentation
  - NEVER uses horizontal flipping (which corrupts character glyphs)
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CRNNDataset")


class CharsetVocab:
    """
    Manages mapping between characters and CTC integer labels.
    73 base classes + space (74) + CTC blank token (index 74 or len(vocab)).
    """

    def __init__(self, config: Dict[str, Any]):
        charset_cfg = config.get("charset", {})
        
        # 1. Base characters in strict order specified in synth_pages README:
        # a-z (0-25), A-Z (26-51), 0-9 (52-61), punctuation (62-72)
        lowercase = charset_cfg.get("lowercase", "abcdefghijklmnopqrstuvwxyz")
        uppercase = charset_cfg.get("uppercase", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        digits = charset_cfg.get("digits", "0123456789")
        punctuation = charset_cfg.get("punctuation", ".,;:!?'\"-()")
        
        chars = list(lowercase) + list(uppercase) + list(digits) + list(punctuation)
        
        # 2. Add space character if configured
        if charset_cfg.get("include_space", True):
            chars.append(" ")

        self.chars = chars
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        
        # 3. CTC Blank symbol is assigned the last index
        self.blank_idx = len(chars)
        self.blank_char = charset_cfg.get("blank_char", "[blank]")
        self.idx_to_char[self.blank_idx] = self.blank_char
        
        # Total classes including CTC blank
        self.num_classes = len(chars) + 1
        
        logger.info(
            f"Initialized vocabulary: {len(chars)} visible characters "
            f"+ 1 CTC blank (idx={self.blank_idx}). Total classes = {self.num_classes}."
        )

    def text_to_indices(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Converts a text string to a list of integer class indices."""
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                # Log or ignore unknown glyphs
                continue
        if max_len is not None:
            indices = indices[:max_len]
        return indices

    def indices_to_text(self, indices: List[int], remove_blank: bool = True) -> str:
        """Converts integer class indices back to text string."""
        chars = []
        for idx in indices:
            if remove_blank and idx == self.blank_idx:
                continue
            if idx in self.idx_to_char and idx != self.blank_idx:
                chars.append(self.idx_to_char[idx])
        return "".join(chars)


def load_labels_df(labels_path: str) -> pd.DataFrame:
    """
    Loads labels.tsv respecting synth_pages Pitfall 1:
      - quoting=csv.QUOTE_NONE
      - keep_default_na=False
      - dtype={'text': str}
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"CRNN labels file not found at: {labels_path}")

    df = pd.read_csv(
        labels_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        dtype={"text": str},
    )
    logger.info(f"Loaded {len(df)} lines from {labels_path}")
    return df


def preprocess_image(
    image_bytes: tf.Tensor,
    target_height: int = 48,
    target_width: int = 800,
    channels: int = 1,
    normalize: bool = True,
    pad_value: float = 255.0
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads and normalizes an image crop:
      - Decodes PNG (handles 1 or 3 channels)
      - Scales width proportionally to keep aspect ratio at height=48
      - Pads the remaining width on the right with white background
      - Returns (padded_image, valid_width)
    """
    # Decode PNG
    img = tf.io.decode_png(image_bytes, channels=3)
    if channels == 1:
        img = tf.image.rgb_to_grayscale(img)

    img = tf.cast(img, tf.float32)
    original_shape = tf.shape(img)
    orig_h = original_shape[0]
    orig_w = original_shape[1]

    # Calculate proportional width for target_height = 48
    scale = tf.cast(target_height, tf.float32) / tf.cast(orig_h, tf.float32)
    new_w = tf.cast(tf.round(tf.cast(orig_w, tf.float32) * scale), tf.int32)
    new_w = tf.maximum(new_w, 1)

    # If scaled width exceeds target_width, resize width down to target_width
    clamped_w = tf.minimum(new_w, target_width)

    # Resize image keeping aspect ratio
    resized_img = tf.image.resize(
        img,
        [target_height, clamped_w],
        method=tf.image.ResizeMethod.BILINEAR
    )

    # Pad on right up to target_width with pad_value (white: 255.0)
    pad_w = target_width - clamped_w
    paddings = [[0, 0], [0, pad_w], [0, 0]]
    padded_img = tf.pad(
        resized_img,
        paddings,
        mode="CONSTANT",
        constant_values=pad_value
    )
    padded_img.set_shape([target_height, target_width, channels])

    # Normalize to [-1.0, 1.0]
    if normalize:
        padded_img = (padded_img / 127.5) - 1.0

    return padded_img, clamped_w


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Subtle photometric augmentations suitable for document OCR text lines:
    - Subtle brightness jitter
    - Subtle contrast jitter
    - CRITICAL: NO horizontal or vertical flipping (which destroys character labels!)
    """
    # Note: image is in [-1.0, 1.0]
    # Brightness adjustment
    delta_brightness = tf.random.uniform([], -0.15, 0.15)
    image = tf.clip_by_value(image + delta_brightness, -1.0, 1.0)
    
    # Contrast adjustment
    contrast_factor = tf.random.uniform([], 0.85, 1.15)
    mean = tf.reduce_mean(image)
    image = tf.clip_by_value((image - mean) * contrast_factor + mean, -1.0, 1.0)

    return image


class CRNNDatasetBuilder:
    """
    Builds optimized tf.data.Dataset objects for CRNN training, validation, and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocab = CharsetVocab(config)
        self.data_root = Path(config["paths"]["data_root"])
        self.labels_path = Path(config["paths"]["labels_file"])

        self.img_h = config["dataset"].get("img_height", 48)
        self.img_w = config["dataset"].get("max_img_width", 800)
        self.channels = config["dataset"].get("channels", 1)
        self.max_text_len = config["dataset"].get("max_text_len", 90)
        self.normalize = config["dataset"].get("normalize", True)
        self.pad_value = float(config["dataset"].get("pad_value", 255.0))
        self.augment_train = config["dataset"].get("augment_train", True)

        # CNN feature extractor downsamples width by 4x
        self.downsample_factor = 4
        self.max_time_steps = self.img_w // self.downsample_factor

        self.labels_df = load_labels_df(str(self.labels_path))

    def get_split_df(self, split_folder: str) -> pd.DataFrame:
        """Filters dataframe by folder name (as stored in 'split' column)."""
        df_split = self.labels_df[self.labels_df["split"] == split_folder].copy()
        # Drop entries where image file does not exist
        valid_rows = []
        for idx, row in df_split.iterrows():
            img_rel = row["image_path"]
            img_full = self.data_root / img_rel
            if img_full.exists():
                valid_rows.append(True)
            else:
                valid_rows.append(False)
        
        filtered = df_split[valid_rows]
        logger.info(
            f"Split '{split_folder}': found {len(filtered)} valid images "
            f"out of {len(df_split)} rows."
        )
        return filtered

    def create_tf_dataset(
        self,
        split_folder: str,
        batch_size: int = 32,
        is_training: bool = False,
        shuffle_buffer: int = 1024,
        limit_samples: Optional[int] = None
    ) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset yielding:
          inputs: {
             'image': tensor (B, H, W, C),
             'label': tensor (B, max_text_len),
             'input_length': tensor (B,),
             'label_length': tensor (B,)
          }
          outputs: dummy label tensor for tf.keras.Model compatibility
        """
        df = self.get_split_df(split_folder)
        if limit_samples:
            df = df.iloc[:limit_samples]

        image_paths = [str(self.data_root / rel) for rel in df["image_path"]]
        texts = [str(t) for t in df["text"]]

        # Pre-encode labels
        label_seqs = []
        label_lengths = []
        valid_indices = []

        for i, text in enumerate(texts):
            indices = self.vocab.text_to_indices(text, max_len=self.max_text_len)
            l_len = len(indices)
            if l_len == 0:
                continue
            
            # Pad to max_text_len with CTC blank symbol
            padded = indices + [self.vocab.blank_idx] * (self.max_text_len - l_len)
            label_seqs.append(padded)
            label_lengths.append(l_len)
            valid_indices.append(i)

        image_paths = [image_paths[i] for i in valid_indices]
        label_seqs = np.array(label_seqs, dtype=np.int32)
        label_lengths = np.array(label_lengths, dtype=np.int32)

        ds = tf.data.Dataset.from_tensor_slices((image_paths, label_seqs, label_lengths))

        if is_training:
            ds = ds.shuffle(buffer_size=min(len(image_paths), shuffle_buffer))

        def _load_and_process(img_path, label, label_len):
            img_bytes = tf.io.read_file(img_path)
            img, valid_w = preprocess_image(
                img_bytes,
                target_height=self.img_h,
                target_width=self.img_w,
                channels=self.channels,
                normalize=self.normalize,
                pad_value=self.pad_value
            )

            if is_training and self.augment_train:
                img = augment_image(img)

            # Feature sequence length for CTC after 4x width downsampling
            valid_steps = tf.maximum(1, valid_w // self.downsample_factor)
            # Ensure valid_steps >= label_len to prevent CTC collapse
            valid_steps = tf.maximum(valid_steps, label_len)
            valid_steps = tf.minimum(valid_steps, self.max_time_steps)

            return {
                "image": img,
                "label": label,
                "input_length": valid_steps,
                "label_length": label_len
            }, label

        num_parallel = self.config["dataset"].get("num_parallel_calls", tf.data.AUTOTUNE)
        ds = ds.map(_load_and_process, num_parallel_calls=num_parallel)
        ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.prefetch(self.config["dataset"].get("prefetch_buffer", tf.data.AUTOTUNE))

        return ds
