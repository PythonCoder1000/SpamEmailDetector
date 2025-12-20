import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from batching import collate_text_batch
from config import CONFIG
from dataset import TextDataset
from metrics import collect_spam_probabilities, compute_confusion_counts_from_threshold, compute_metrics_from_counts, select_decision_threshold
from model import TransformerTextClassifier
from text_processing import build_vocabulary
from train_utils import load_best_decision_threshold, load_best_model_weights, save_artifacts, set_random_seed, train_one_epoch


def resolve_dataset_path(datasets_directory: str, csv_file_name: str) -> str:
    if os.path.isabs(csv_file_name):
        return csv_file_name
    return os.path.join(datasets_directory, csv_file_name)


def load_dataframe() -> pd.DataFrame:
    dataset_path = resolve_dataset_path(CONFIG.datasets_directory, CONFIG.csv_file_name)
    full_dataframe = pd.read_csv(dataset_path)

    if CONFIG.text_column_name not in full_dataframe.columns or CONFIG.label_column_name not in full_dataframe.columns:
        raise ValueError(
            f"CSV must contain columns {CONFIG.text_column_name!r} and {CONFIG.label_column_name!r}. Found: {list(full_dataframe.columns)}"
        )

    working_dataframe = full_dataframe[[CONFIG.text_column_name, CONFIG.label_column_name]].copy()
    working_dataframe[CONFIG.text_column_name] = working_dataframe[CONFIG.text_column_name].astype(str)
    working_dataframe[CONFIG.label_column_name] = working_dataframe[CONFIG.label_column_name].astype(int)
    working_dataframe = working_dataframe.rename(columns={CONFIG.text_column_name: "text", CONFIG.label_column_name: "spam"})

    return working_dataframe


def split_dataframe(full_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_full = full_dataframe["spam"].values
    remaining_dataframe, test_dataframe = train_test_split(
        full_dataframe,
        test_size=CONFIG.test_fraction,
        random_state=CONFIG.random_seed,
        stratify=labels_full,
        shuffle=True,
    )

    labels_remaining = remaining_dataframe["spam"].values
    validation_dataframe, training_dataframe = train_test_split(
        remaining_dataframe,
        test_size=1.0 - CONFIG.validation_fraction_of_remaining,
        random_state=CONFIG.random_seed,
        stratify=labels_remaining,
        shuffle=True,
    )

    return training_dataframe, validation_dataframe, test_dataframe


def get_early_stopping_metric_value(validation_metrics: dict) -> float:
    supported_metric_values = {
        "val_accuracy": float(validation_metrics["accuracy"]),
        "val_spam_precision": float(validation_metrics["spam_precision"]),
        "val_spam_recall": float(validation_metrics["spam_recall"]),
        "val_spam_f1": float(validation_metrics["spam_f1"]),
        "val_false_positive_rate": float(validation_metrics["false_positive_rate"]),
    }
    if CONFIG.early_stopping_metric not in supported_metric_values:
        raise ValueError(f"Unknown early_stopping_metric={CONFIG.early_stopping_metric!r}. Available: {sorted(supported_metric_values.keys())}")
    return supported_metric_values[CONFIG.early_stopping_metric]


def main() -> None:
    set_random_seed(CONFIG.random_seed)
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataframe = load_dataframe()
    training_dataframe, validation_dataframe, test_dataframe = split_dataframe(full_dataframe)

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

    best_early_stopping_metric_value = -1e30
    best_decision_threshold = 0.5
    epochs_without_improvement = 0

    for epoch_index in range(1, CONFIG.epoch_count + 1):
        average_training_loss = train_one_epoch(
            model=model,
            data_loader=training_data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            gradient_clipping_norm=CONFIG.gradient_clipping_norm,
        )

        validation_probabilities, validation_labels = collect_spam_probabilities(model, validation_data_loader)
        selected_threshold, validation_metrics = select_decision_threshold(
            probabilities=validation_probabilities,
            labels=validation_labels,
            decision_threshold_grid_points=CONFIG.decision_threshold_grid_points,
            minimum_spam_recall=CONFIG.minimum_spam_recall_for_threshold_selection,
            threshold_selection_strategy=CONFIG.threshold_selection_strategy,
        )

        validation_metrics_with_threshold = dict(validation_metrics)
        validation_metrics_with_threshold["decision_threshold"] = float(selected_threshold)

        early_stopping_metric_value = get_early_stopping_metric_value(validation_metrics_with_threshold)
        improved = early_stopping_metric_value > (best_early_stopping_metric_value + CONFIG.early_stopping_min_delta)

        if improved:
            best_early_stopping_metric_value = early_stopping_metric_value
            best_decision_threshold = float(selected_threshold)
            epochs_without_improvement = 0
            save_artifacts(
                model=model,
                token_to_index=token_to_index,
                config=CONFIG,
                best_decision_threshold=best_decision_threshold,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch_index} "
            f"training_loss={average_training_loss:.4f} "
            f"val_threshold={validation_metrics_with_threshold['decision_threshold']:.3f} "
            f"val_accuracy={validation_metrics_with_threshold['accuracy']:.4f} "
            f"val_spam_precision={validation_metrics_with_threshold['spam_precision']:.4f} "
            f"val_spam_recall={validation_metrics_with_threshold['spam_recall']:.4f} "
            f"val_spam_f1={validation_metrics_with_threshold['spam_f1']:.4f} "
            f"val_false_positive_rate={validation_metrics_with_threshold['false_positive_rate']:.4f} "
            f"val_confusion_matrix={validation_metrics_with_threshold['confusion_matrix']} "
            f"best_{CONFIG.early_stopping_metric}={best_early_stopping_metric_value:.4f}"
        )

        if CONFIG.early_stopping_enabled and epochs_without_improvement >= CONFIG.early_stopping_patience:
            print(
                f"early_stopping_triggered=True "
                f"epoch={epoch_index} "
                f"patience={CONFIG.early_stopping_patience} "
                f"metric={CONFIG.early_stopping_metric} "
                f"best_metric={best_early_stopping_metric_value:.4f}"
            )
            break

    load_best_model_weights(model, CONFIG.models_directory, CONFIG.saved_model_name, compute_device)
    best_decision_threshold = load_best_decision_threshold(CONFIG.models_directory, CONFIG.saved_threshold_name)

    test_probabilities, test_labels = collect_spam_probabilities(model, test_data_loader)
    test_counts = compute_confusion_counts_from_threshold(test_probabilities, test_labels, best_decision_threshold)
    test_metrics = compute_metrics_from_counts(*test_counts)

    print(
        f"test_threshold={best_decision_threshold:.3f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_spam_precision={test_metrics['spam_precision']:.4f} "
        f"test_spam_recall={test_metrics['spam_recall']:.4f} "
        f"test_spam_f1={test_metrics['spam_f1']:.4f} "
        f"test_false_positive_rate={test_metrics['false_positive_rate']:.4f} "
        f"test_confusion_matrix={test_metrics['confusion_matrix']}"
    )


if __name__ == "__main__":
    main()
