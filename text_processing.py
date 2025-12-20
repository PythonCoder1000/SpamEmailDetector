import re
from collections import Counter


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
