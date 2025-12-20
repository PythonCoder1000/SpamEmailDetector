import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def collect_spam_probabilities(model: nn.Module, data_loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_values = []
    label_values = []

    for input_id_tensor, attention_mask_tensor, label_tensor in data_loader:
        logit_tensor = model(input_id_tensor, attention_mask_tensor)
        probability_tensor = torch.softmax(logit_tensor, dim=1)[:, 1]
        probability_values.append(probability_tensor.detach().cpu().numpy())
        label_values.append(label_tensor.detach().cpu().numpy())

    probabilities = np.concatenate(probability_values, axis=0) if probability_values else np.array([], dtype=np.float32)
    labels = np.concatenate(label_values, axis=0) if label_values else np.array([], dtype=np.int64)
    return probabilities, labels


def compute_confusion_counts_from_threshold(probabilities: np.ndarray, labels: np.ndarray, decision_threshold: float) -> tuple[int, int, int, int]:
    predictions = (probabilities >= decision_threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    true_negative_count = int(np.sum((predictions == 0) & (labels == 0)))
    false_positive_count = int(np.sum((predictions == 1) & (labels == 0)))
    false_negative_count = int(np.sum((predictions == 0) & (labels == 1)))
    true_positive_count = int(np.sum((predictions == 1) & (labels == 1)))
    return true_negative_count, false_positive_count, false_negative_count, true_positive_count


def compute_metrics_from_counts(true_negative_count: int, false_positive_count: int, false_negative_count: int, true_positive_count: int) -> dict:
    total_count = true_negative_count + false_positive_count + false_negative_count + true_positive_count
    accuracy_value = (true_negative_count + true_positive_count) / max(1, total_count)

    precision_denominator = true_positive_count + false_positive_count
    recall_denominator = true_positive_count + false_negative_count

    spam_precision_value = true_positive_count / max(1, precision_denominator)
    spam_recall_value = true_positive_count / max(1, recall_denominator)

    f1_denominator = spam_precision_value + spam_recall_value
    spam_f1_value = (2.0 * spam_precision_value * spam_recall_value / f1_denominator) if f1_denominator > 0 else 0.0

    false_positive_rate_denominator = false_positive_count + true_negative_count
    false_positive_rate_value = false_positive_count / max(1, false_positive_rate_denominator)

    return {
        "accuracy": float(accuracy_value),
        "spam_precision": float(spam_precision_value),
        "spam_recall": float(spam_recall_value),
        "spam_f1": float(spam_f1_value),
        "false_positive_rate": float(false_positive_rate_value),
        "confusion_matrix": [[int(true_negative_count), int(false_positive_count)], [int(false_negative_count), int(true_positive_count)]],
    }


def select_decision_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    decision_threshold_grid_points: int,
    minimum_spam_recall: float,
    threshold_selection_strategy: str,
) -> tuple[float, dict]:
    if probabilities.size == 0:
        default_threshold = 0.5
        default_counts = compute_confusion_counts_from_threshold(probabilities, labels, default_threshold)
        return default_threshold, compute_metrics_from_counts(*default_counts)

    threshold_values = np.linspace(0.05, 0.95, max(2, int(decision_threshold_grid_points))).astype(np.float64)

    best_threshold = 0.5
    best_metrics = None

    best_precision = -1.0
    best_false_positive_rate = float("inf")
    best_f1 = -1.0

    for threshold_value in threshold_values:
        counts = compute_confusion_counts_from_threshold(probabilities, labels, float(threshold_value))
        metrics = compute_metrics_from_counts(*counts)

        meets_recall = metrics["spam_recall"] >= float(minimum_spam_recall)

        if threshold_selection_strategy == "maximize_precision_with_min_recall":
            if meets_recall:
                if (metrics["spam_precision"] > best_precision) or (
                    metrics["spam_precision"] == best_precision and metrics["false_positive_rate"] < best_false_positive_rate
                ):
                    best_precision = metrics["spam_precision"]
                    best_false_positive_rate = metrics["false_positive_rate"]
                    best_threshold = float(threshold_value)
                    best_metrics = metrics
        elif threshold_selection_strategy == "minimize_false_positive_rate_with_min_recall":
            if meets_recall:
                if (metrics["false_positive_rate"] < best_false_positive_rate) or (
                    metrics["false_positive_rate"] == best_false_positive_rate and metrics["spam_precision"] > best_precision
                ):
                    best_precision = metrics["spam_precision"]
                    best_false_positive_rate = metrics["false_positive_rate"]
                    best_threshold = float(threshold_value)
                    best_metrics = metrics
        else:
            if metrics["spam_f1"] > best_f1:
                best_f1 = metrics["spam_f1"]
                best_threshold = float(threshold_value)
                best_metrics = metrics

    if best_metrics is None:
        for threshold_value in threshold_values:
            counts = compute_confusion_counts_from_threshold(probabilities, labels, float(threshold_value))
            metrics = compute_metrics_from_counts(*counts)
            if metrics["spam_f1"] > best_f1:
                best_f1 = metrics["spam_f1"]
                best_threshold = float(threshold_value)
                best_metrics = metrics

    return best_threshold, best_metrics
