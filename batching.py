import torch

from text_processing import encode_text_to_token_ids


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
