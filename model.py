import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextDataset


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
    epoch_count: int = 20
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

    early_stopping_enabled: bool = True
    early_stopping_metric: str = "spam_f1"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4


CONFIG = TrainingConfig()


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


token_pattern = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize_text(raw_text: str) -> list[str]:
    if raw_text is None:
        raw_text = ""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    return token_pattern.findall(raw_text.lower())


def build_vocabulary(training_texts: list[str], maximum_vocabulary_size: int, minimum_token_frequency: int) -> dict[str, int]:
    token_counter = Counter()
    for training_text in training_texts:
        token_counter.update(tokenize_text(training_text))

    token_to_index = {"<pad>": 0, "<unk>": 1, "<cls>": 2}

    for token, token_frequency in token_counter.most_common():
        if token_frequency < minimum_token_frequency:
            continue
        if token in token_to_index:
            continue
        token_to_index[token] = len(token_to_index)
        if len(token_to_index) >= maximum_vocabulary_size:
            break

    return token_to_index


def encode_text_to_token_ids(raw_text: str, token_to_index: dict[str, int], maximum_sequence_length: int) -> list[int]:
    padding_token_id = token_to_index["<pad>"]
    unknown_token_id = token_to_index["<unk>"]
    classification_token_id = token_to_index["<cls>"]

    token_ids = [classification_token_id]
    for token in tokenize_text(raw_text):
        token_ids.append(token_to_index.get(token, unknown_token_id))
        if len(token_ids) >= maximum_sequence_length:
            break

    if len(token_ids) < maximum_sequence_length:
        token_ids.extend([padding_token_id] * (maximum_sequence_length - len(token_ids)))

    return token_ids


def collate_text_batch(
    batch_samples: list[tuple[str, int]],
    token_to_index: dict[str, int],
    maximum_sequence_length: int,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded_input_id_rows = []
    label_values = []

    for raw_text, label_value in batch_samples:
        encoded_input_id_rows.append(encode_text_to_token_ids(raw_text, token_to_index, maximum_sequence_length))
        label_values.append(int(label_value))

    input_id_tensor = torch.tensor(encoded_input_id_rows, dtype=torch.long, device=compute_device)
    label_tensor = torch.tensor(label_values, dtype=torch.long, device=compute_device)

    padding_token_id = token_to_index["<pad>"]
    attention_mask_tensor = (input_id_tensor != padding_token_id)

    return input_id_tensor, attention_mask_tensor, label_tensor


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


def resolve_dataset_path(datasets_directory: str, csv_file_name: str) -> str:
    if os.path.isabs(csv_file_name):
        return csv_file_name
    return os.path.join(datasets_directory, csv_file_name)


def load_dataframe(config: TrainingConfig) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(config.datasets_directory, config.csv_file_name)
    full_dataframe = pd.read_csv(dataset_path)

    if config.text_column_name not in full_dataframe.columns or config.label_column_name not in full_dataframe.columns:
        raise ValueError(
            f"CSV must contain columns {config.text_column_name!r} and {config.label_column_name!r}. Found: {list(full_dataframe.columns)}"
        )

    working_dataframe = full_dataframe[[config.text_column_name, config.label_column_name]].copy()
    working_dataframe[config.text_column_name] = working_dataframe[config.text_column_name].astype(str)
    working_dataframe[config.label_column_name] = working_dataframe[config.label_column_name].astype(int)
    working_dataframe = working_dataframe.rename(columns={config.text_column_name: "text", config.label_column_name: "spam"})

    return working_dataframe


def split_dataframe(
    full_dataframe: pd.DataFrame,
    test_fraction: float,
    validation_fraction_of_remaining: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_full = full_dataframe["spam"].values
    remaining_dataframe, test_dataframe = train_test_split(
        full_dataframe,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=labels_full,
        shuffle=True,
    )

    labels_remaining = remaining_dataframe["spam"].values
    validation_dataframe, training_dataframe = train_test_split(
        remaining_dataframe,
        test_size=1.0 - validation_fraction_of_remaining,
        random_state=random_seed,
        stratify=labels_remaining,
        shuffle=True,
    )

    return training_dataframe, validation_dataframe, test_dataframe


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader) -> dict:
    model.eval()
    all_predictions = []
    all_labels = []

    for input_id_tensor, attention_mask_tensor, label_tensor in data_loader:
        logit_tensor = model(input_id_tensor, attention_mask_tensor)
        predicted_class_tensor = torch.argmax(logit_tensor, dim=1)
        all_predictions.extend(predicted_class_tensor.detach().cpu().tolist())
        all_labels.extend(label_tensor.detach().cpu().tolist())

    accuracy_value = accuracy_score(all_labels, all_predictions)
    precision_value, recall_value, f1_value, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    confusion_matrix_values = confusion_matrix(all_labels, all_predictions).tolist()

    return {
        "accuracy": float(accuracy_value),
        "spam_precision": float(precision_value),
        "spam_recall": float(recall_value),
        "spam_f1": float(f1_value),
        "confusion_matrix": confusion_matrix_values,
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    gradient_clipping_norm: float,
) -> float:
    model.train()
    running_loss_sum = 0.0
    running_sample_count = 0

    progress_bar = tqdm(data_loader, desc="training", leave=False)
    for input_id_tensor, attention_mask_tensor, label_tensor in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        logit_tensor = model(input_id_tensor, attention_mask_tensor)
        loss_value = loss_function(logit_tensor, label_tensor)
        loss_value.backward()

        if gradient_clipping_norm and gradient_clipping_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

        optimizer.step()

        batch_size = label_tensor.size(0)
        running_loss_sum += float(loss_value.item()) * batch_size
        running_sample_count += batch_size
        average_loss_value = running_loss_sum / max(1, running_sample_count)
        progress_bar.set_postfix(loss=f"{average_loss_value:.4f}")

    return running_loss_sum / max(1, running_sample_count)


def ensure_directory_exists(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def save_artifacts(
    model: nn.Module,
    token_to_index: dict[str, int],
    config: TrainingConfig,
) -> None:
    ensure_directory_exists(config.models_directory)

    model_path = os.path.join(config.models_directory, config.saved_model_name)
    vocabulary_path = os.path.join(config.models_directory, config.saved_vocabulary_name)
    config_path = os.path.join(config.models_directory, config.saved_config_name)

    torch.save(model.state_dict(), model_path)
    with open(vocabulary_path, "w", encoding="utf-8") as file_handle:
        json.dump(token_to_index, file_handle, ensure_ascii=False)
    with open(config_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(config), file_handle, ensure_ascii=False)


def get_metric_value(metrics: dict, metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric {metric_name!r}. Available: {sorted(metrics.keys())}")
    return float(metrics[metric_name])


def main() -> None:
    set_random_seed(CONFIG.random_seed)
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataframe = load_dataframe(CONFIG)
    training_dataframe, validation_dataframe, test_dataframe = split_dataframe(
        full_dataframe=full_dataframe,
        test_fraction=CONFIG.test_fraction,
        validation_fraction_of_remaining=CONFIG.validation_fraction_of_remaining,
        random_seed=CONFIG.random_seed,
    )

    training_dataset = TextDataset(training_dataframe)
    validation_dataset = TextDataset(validation_dataframe)
    test_dataset = TextDataset(test_dataframe)

    training_texts = training_dataframe["text"].tolist()
    token_to_index = build_vocabulary(
        training_texts=training_texts,
        maximum_vocabulary_size=CONFIG.maximum_vocabulary_size,
        minimum_token_frequency=CONFIG.minimum_token_frequency,
    )

    padding_token_id = token_to_index["<pad>"]

    model = TransformerTextClassifier(
        vocabulary_size=len(token_to_index),
        maximum_sequence_length=CONFIG.maximum_sequence_length,
        model_dimension=CONFIG.model_dimension,
        attention_head_count=CONFIG.attention_head_count,
        encoder_layer_count=CONFIG.encoder_layer_count,
        feedforward_dimension=CONFIG.feedforward_dimension,
        dropout_probability=CONFIG.dropout_probability,
        padding_token_id=padding_token_id,
        class_count=2,
    ).to(compute_device)

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.data_loader_worker_count,
        collate_fn=lambda batch_samples: collate_text_batch(
            batch_samples=batch_samples,
            token_to_index=token_to_index,
            maximum_sequence_length=CONFIG.maximum_sequence_length,
            compute_device=compute_device,
        ),
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.data_loader_worker_count,
        collate_fn=lambda batch_samples: collate_text_batch(
            batch_samples=batch_samples,
            token_to_index=token_to_index,
            maximum_sequence_length=CONFIG.maximum_sequence_length,
            compute_device=compute_device,
        ),
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.data_loader_worker_count,
        collate_fn=lambda batch_samples: collate_text_batch(
            batch_samples=batch_samples,
            token_to_index=token_to_index,
            maximum_sequence_length=CONFIG.maximum_sequence_length,
            compute_device=compute_device,
        ),
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)

    best_validation_metric_value = -1e30
    epochs_without_improvement = 0

    for epoch_index in range(1, CONFIG.epoch_count + 1):
        average_training_loss = train_one_epoch(
            model=model,
            data_loader=training_data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            gradient_clipping_norm=CONFIG.gradient_clipping_norm,
        )

        validation_metrics = evaluate_model(model, validation_data_loader)
        validation_metric_value = get_metric_value(validation_metrics, CONFIG.early_stopping_metric)

        improved = validation_metric_value > (best_validation_metric_value + CONFIG.early_stopping_min_delta)
        if improved:
            best_validation_metric_value = validation_metric_value
            epochs_without_improvement = 0
            save_artifacts(model=model, token_to_index=token_to_index, config=CONFIG)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch_index} "
            f"training_loss={average_training_loss:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f} "
            f"val_spam_precision={validation_metrics['spam_precision']:.4f} "
            f"val_spam_recall={validation_metrics['spam_recall']:.4f} "
            f"val_spam_f1={validation_metrics['spam_f1']:.4f} "
            f"val_confusion_matrix={validation_metrics['confusion_matrix']} "
            f"best_val_{CONFIG.early_stopping_metric}={best_validation_metric_value:.4f}"
        )

        if CONFIG.early_stopping_enabled and epochs_without_improvement >= CONFIG.early_stopping_patience:
            print(
                f"early_stopping_triggered=True "
                f"epoch={epoch_index} "
                f"patience={CONFIG.early_stopping_patience} "
                f"metric={CONFIG.early_stopping_metric} "
                f"best_metric={best_validation_metric_value:.4f}"
            )
            break

    test_metrics = evaluate_model(model, test_data_loader)
    print(
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_spam_precision={test_metrics['spam_precision']:.4f} "
        f"test_spam_recall={test_metrics['spam_recall']:.4f} "
        f"test_spam_f1={test_metrics['spam_f1']:.4f} "
        f"test_confusion_matrix={test_metrics['confusion_matrix']}"
    )


if __name__ == "__main__":
    main()
