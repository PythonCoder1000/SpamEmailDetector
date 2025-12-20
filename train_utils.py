import json
import os
import random
from dataclasses import asdict

import torch
import torch.nn as nn
from tqdm import tqdm


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directory_exists(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def save_artifacts(model: nn.Module, token_to_index: dict[str, int], config, best_decision_threshold: float) -> None:
    ensure_directory_exists(config.models_directory)

    model_path = os.path.join(config.models_directory, config.saved_model_name)
    vocabulary_path = os.path.join(config.models_directory, config.saved_vocabulary_name)
    config_path = os.path.join(config.models_directory, config.saved_config_name)
    threshold_path = os.path.join(config.models_directory, config.saved_threshold_name)

    torch.save(model.state_dict(), model_path)

    with open(vocabulary_path, "w", encoding="utf-8") as file_handle:
        json.dump(token_to_index, file_handle, ensure_ascii=False)

    with open(config_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(config), file_handle, ensure_ascii=False)

    with open(threshold_path, "w", encoding="utf-8") as file_handle:
        json.dump({"decision_threshold": float(best_decision_threshold)}, file_handle, ensure_ascii=False)


def load_best_model_weights(model: nn.Module, models_directory: str, saved_model_name: str, compute_device: torch.device) -> None:
    best_model_path = os.path.join(models_directory, saved_model_name)
    state_dictionary = torch.load(best_model_path, map_location=compute_device)
    model.load_state_dict(state_dictionary)
    model.to(compute_device)


def load_best_decision_threshold(models_directory: str, saved_threshold_name: str) -> float:
    threshold_path = os.path.join(models_directory, saved_threshold_name)
    with open(threshold_path, "r", encoding="utf-8") as file_handle:
        threshold_object = json.load(file_handle)
    return float(threshold_object["decision_threshold"])


def train_one_epoch(
    model: nn.Module,
    data_loader,
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
