from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    datasets_directory: str = "datasets"
    csv_file_name: str = "emails.csv"
    text_column_name: str = "text"
    label_column_name: str = "spam"

    test_fraction: float = 0.2
    validation_fraction_of_remaining: float = 0.1
    random_seed: int = 42

    maximum_sequence_length: int = 160
    maximum_vocabulary_size: int = 30000
    minimum_token_frequency: int = 2

    batch_size: int = 64
    epoch_count: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clipping_norm: float = 1.0

    model_dimension: int = 128
    attention_head_count: int = 4
    encoder_layer_count: int = 2
    feedforward_dimension: int = 256
    dropout_probability: float = 0.1

    data_loader_worker_count: int = 0

    models_directory: str = "models"
    saved_model_name: str = "spam_transformer_best.pt"
    saved_vocabulary_name: str = "spam_transformer_vocab.json"
    saved_config_name: str = "spam_transformer_config.json"
    saved_threshold_name: str = "spam_transformer_threshold.json"

    early_stopping_enabled: bool = True
    early_stopping_metric: str = "val_spam_precision"
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4

    decision_threshold_grid_points: int = 41
    minimum_spam_recall_for_threshold_selection: float = 0.80
    threshold_selection_strategy: str = "maximize_precision_with_min_recall"


CONFIG = TrainingConfig()
