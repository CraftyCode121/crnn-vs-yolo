"""
model.py - Convolutional Recurrent Neural Network (CRNN) with CTC Loss for OCR.
Implements:
  - VGG-style CNN feature extractor with width downsampling factor of 4
  - Map-to-sequence transformation
  - 2-layer Stacked Bidirectional LSTM sequence modeling
  - CTC Loss handling via tf.nn.ctc_loss with gradient clipping
  - Greedy and Beam Search CTC decoding
"""

from typing import Dict, Any, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model


class CRNNModel(Model):
    """
    Complete CRNN Architecture subclassed with built-in CTC Loss and metrics.
    """

    def __init__(
        self,
        img_height: int = 48,
        img_width: int = 800,
        channels: int = 1,
        num_classes: int = 75,
        blank_idx: int = 74,
        rnn_units: int = 256,
        rnn_layers_count: int = 2,
        dropout_rate: float = 0.25,
        fc_units: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.num_classes = num_classes
        self.blank_idx = blank_idx
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate

        # --- 1. CNN Backbone (VGG-style for text lines) ---
        # Stage 1: 48 -> 24 (H), W -> W/2
        self.conv1 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1")

        # Stage 2: 24 -> 12 (H), W/2 -> W/4
        self.conv2 = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2")

        # Stage 3: 12 -> 6 (H), W/4 -> W/4 (strides (2, 1) keeps width resolution)
        self.conv3 = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3")
        self.bn3 = layers.BatchNormalization(name="bn3")
        self.conv4 = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="pool3")

        # Stage 4: 6 -> 3 (H), W/4 -> W/4
        self.conv5 = layers.Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5")
        self.bn5 = layers.BatchNormalization(name="bn5")
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="pool4")

        # Stage 5: 3 -> 1 (H), W/4 -> W/4 (kernel (3, 1) collapses remaining height)
        self.conv6 = layers.Conv2D(512, (3, 1), padding="valid", activation="relu", name="conv6")
        self.bn6 = layers.BatchNormalization(name="bn6")

        # --- 2. Sequence Projection ---
        self.fc_proj = layers.Dense(fc_units, activation="relu", name="fc_projection")
        self.drop_fc = layers.Dropout(dropout_rate, name="drop_fc")

        # --- 3. Bidirectional RNN Stack ---
        self.rnn_layers = []
        for i in range(rnn_layers_count):
            self.rnn_layers.append(
                layers.Bidirectional(
                    layers.LSTM(
                        rnn_units,
                        return_sequences=True,
                        dropout=dropout_rate,
                        recurrent_dropout=0.0  # CuDNN compatible
                    ),
                    name=f"bilstm_{i+1}"
                )
            )

        # --- 4. Final CTC Projection ---
        # Outputs unnormalized logits for num_classes (including CTC blank token)
        self.logits_dense = layers.Dense(num_classes, activation=None, name="logits_dense")

        # Tracking metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def call(self, inputs, training=False):
        # 1. Feature Extraction
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.pool3(x)

        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.pool4(x)

        x = self.conv6(x)
        x = self.bn6(x, training=training)

        # x shape: (B, 1, seq_len, 512)
        # Squeeze height dimension -> (B, seq_len, 512)
        x = tf.squeeze(x, axis=1)

        # Bottleneck projection
        x = self.fc_proj(x)
        if training:
            x = self.drop_fc(x, training=training)

        # 2. Recurrent Sequence Modeling
        for rnn in self.rnn_layers:
            x = rnn(x, training=training)

        # 3. Logits for CTC
        logits = self.logits_dense(x)
        return logits

    def compute_ctc_loss(
        self,
        labels: tf.Tensor,
        logits: tf.Tensor,
        label_length: tf.Tensor,
        logit_length: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes Connectionist Temporal Classification (CTC) loss.
        Handles padding, blank token index, and masks non-finite values for stability.
        """
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=self.blank_idx
        )
        
        # Mask out NaNs or Infs that could occur with degenerate alignments
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        return tf.reduce_mean(loss)

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

    def train_step(self, data):
        # Unpack data
        inputs, _ = data
        images = inputs["image"]
        labels = inputs["label"]
        input_length = inputs["input_length"]
        label_length = inputs["label_length"]

        with tf.GradientTape() as tape:
            logits = self(images, training=True)
            loss = self.compute_ctc_loss(labels, logits, label_length, input_length)

        # Compute and clip gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Clip by global norm to prevent exploding gradients in LSTMs
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        inputs, _ = data
        images = inputs["image"]
        labels = inputs["label"]
        input_length = inputs["input_length"]
        label_length = inputs["label_length"]

        logits = self(images, training=False)
        loss = self.compute_ctc_loss(labels, logits, label_length, input_length)

        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}

    def decode(
        self,
        logits: tf.Tensor,
        input_length: tf.Tensor,
        method: str = "greedy",
        beam_width: int = 10
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Decodes CTC output logits into sequence predictions.
        Returns:
          decoded_sparse: tf.SparseTensor containing class predictions
          log_probabilities: confidence tensor
        """
        # tf.nn.ctc_*_decoder expects time-major logits: (time, batch, num_classes)
        time_major_logits = tf.transpose(logits, [1, 0, 2])

        if method == "beam_search":
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(
                time_major_logits,
                sequence_length=input_length,
                beam_width=beam_width,
                top_paths=1
            )
            return decoded[0], log_prob
        else:
            decoded, log_prob = tf.nn.ctc_greedy_decoder(
                time_major_logits,
                sequence_length=input_length
            )
            return decoded[0], log_prob


def build_crnn_model(config: Dict[str, Any], num_classes: int, blank_idx: int) -> CRNNModel:
    """
    Factory helper to initialize CRNN model from configuration dictionary.
    """
    ds_cfg = config.get("dataset", {})
    md_cfg = config.get("model", {})

    img_h = ds_cfg.get("img_height", 48)
    img_w = ds_cfg.get("max_img_width", 800)
    channels = ds_cfg.get("channels", 1)

    rnn_units = md_cfg.get("rnn_units", 256)
    rnn_layers_count = md_cfg.get("rnn_layers", 2)
    dropout_rate = md_cfg.get("dropout_rate", 0.25)
    fc_units = md_cfg.get("fc_units", 128)

    model = CRNNModel(
        img_height=img_h,
        img_width=img_w,
        channels=channels,
        num_classes=num_classes,
        blank_idx=blank_idx,
        rnn_units=rnn_units,
        rnn_layers_count=rnn_layers_count,
        dropout_rate=dropout_rate,
        fc_units=fc_units,
        name="crnn_ocr_model"
    )

    # Build model by calling on dummy input tensor
    dummy_input = tf.zeros((1, img_h, img_w, channels), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    return model
