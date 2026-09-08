import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .layers import Projection, ClassificationHead


class ModelB(nn.Module):
    """
    ModelB - Isolated Sign Recognition

    Input:
        keypoints: (batch, time, joints, 2)
        attention_mask: (batch, time)

    Output:
        logits: (batch, num_labels)

    The number of classes is controlled by the configuration:
        num_labels: 100
        num_labels: 300
        num_labels: 1000
        num_labels: 2000

    ModelB is responsible only for isolated sign/word recognition.
    """

    def __init__(self, config):
        super().__init__()

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.joint_idx = config["joint_idx"]

        # ---------------------------------------------------------
        # Model components
        # ---------------------------------------------------------

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.projection = Projection(config)

        # ---------------------------------------------------------
        # Classification head
        #
        # IMPORTANT:
        # num_labels comes from the YAML configuration.
        #
        # WLASL-100  -> 100 classes
        # WLASL-300  -> 300 classes
        # WLASL-1000 -> 1000 classes
        # WLASL-2000 -> 2000 classes
        # ---------------------------------------------------------

        self.classification_head = ClassificationHead(
            config["d_model"],
            config["num_labels"],
            config["classifier_dropout"]
        )

        # ---------------------------------------------------------
        # Loss function
        # ---------------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, keypoints, attention_mask, labels=None):
        """
        Forward pass.

        Args:
            keypoints:
                Tensor of shape (B, T, J, 2)

            attention_mask:
                Tensor of shape (B, T)
                1 = valid frame
                0 = padding

            labels:
                Optional tensor of shape (B,)

        Returns:
            loss:
                Cross-entropy loss if labels are provided,
                otherwise None.

            logits:
                Tensor of shape (B, num_labels)
        """

        # ---------------------------------------------------------
        # Batch size
        # ---------------------------------------------------------

        batch_size = keypoints.shape[0]

        # ---------------------------------------------------------
        # Select the joints required by ModelB
        #
        # Original input:
        #     (B, T, 75, 2)
        #
        # After joint selection:
        #     (B, T, selected_joints, 2)
        # ---------------------------------------------------------

        keypoints = keypoints[:, :, self.joint_idx, :]

        # ---------------------------------------------------------
        # Project keypoints into model embeddings
        # ---------------------------------------------------------

        x_embed, y_embed = self.projection(keypoints)

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        encoder_outputs = self.encoder(
            x_embed=x_embed,
            attention_mask=attention_mask
        )

        # ---------------------------------------------------------
        # Decoder
        #
        # ModelB uses the same temporal attention mask for the
        # decoder because it operates on the same input sequence.
        # ---------------------------------------------------------

        decoder_attention_mask = attention_mask

        decoder_outputs = self.decoder(
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            attention_mask=decoder_attention_mask,
            y_embed=y_embed
        )

        # ---------------------------------------------------------
        # Find the final valid timestep for every sample
        # ---------------------------------------------------------

        last_indices = (
            (decoder_attention_mask == 1)
            .float()
            .cumsum(dim=1)
            .argmax(dim=1)
        )

        # ---------------------------------------------------------
        # Extract the decoder representation at the final
        # valid timestep.
        # ---------------------------------------------------------

        last_decoder_outputs = decoder_outputs[
            torch.arange(batch_size, device=decoder_outputs.device),
            last_indices,
            :
        ]

        # ---------------------------------------------------------
        # Classification
        #
        # Output:
        #     (B, num_labels)
        # ---------------------------------------------------------

        logits = self.classification_head(last_decoder_outputs)

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------

        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.compute_loss(logits, labels)
        else:
            loss = None

        return loss, logits

    def compute_loss(self, logits, labels):
        """
        Calculate classification loss.
        """

        return self.loss_fn(logits, labels)