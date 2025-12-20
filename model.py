import torch
import torch.nn as nn


class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        maximum_sequence_length: int,
        model_dimension: int,
        attention_head_count: int,
        encoder_layer_count: int,
        feedforward_dimension: int,
        dropout_probability: float,
        padding_token_id: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, model_dimension, padding_idx=padding_token_id)
        self.position_embedding = nn.Embedding(maximum_sequence_length, model_dimension)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_head_count,
            dim_feedforward=feedforward_dimension,
            dropout=dropout_probability,
            batch_first=True,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=encoder_layer_count,
            enable_nested_tensor=False,
        )

        self.dropout_layer = nn.Dropout(dropout_probability)
        self.classification_head = nn.Linear(model_dimension, class_count)

    def forward(self, input_id_tensor: torch.Tensor, attention_mask_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_id_tensor.shape
        position_index_tensor = torch.arange(sequence_length, device=input_id_tensor.device).unsqueeze(0).expand(batch_size, sequence_length)

        embedded_token_tensor = self.token_embedding(input_id_tensor)
        embedded_position_tensor = self.position_embedding(position_index_tensor)
        encoder_input_tensor = embedded_token_tensor + embedded_position_tensor

        key_padding_mask_tensor = ~attention_mask_tensor
        encoder_output_tensor = self.transformer_encoder(encoder_input_tensor, src_key_padding_mask=key_padding_mask_tensor)

        classification_token_tensor = encoder_output_tensor[:, 0, :]
        classification_token_tensor = self.dropout_layer(classification_token_tensor)
        logit_tensor = self.classification_head(classification_token_tensor)
        return logit_tensor
