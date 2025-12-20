import os
import re
import json
import csv
import torch
import torch.nn as nn

CSV_PATH = os.path.join("emls", "eml_processed.csv")

MODEL_DIR = "models"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "spam_transformer_best.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "spam_transformer_vocab.json")
THRESH_PATH = os.path.join(MODEL_DIR, "spam_transformer_threshold.json")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]", re.UNICODE)

def _tokenize(text):
    s = "" if text is None else str(text)
    if len(s) > 200000:
        s = s[:200000]
    return _TOKEN_RE.findall(s.lower())

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _coerce_vocab(obj):
    if isinstance(obj, dict) and "vocab" in obj and isinstance(obj["vocab"], dict):
        obj = obj["vocab"]
    if not isinstance(obj, dict) or len(obj) == 0:
        return None

    keys_are_int = True
    vals_are_int = True
    for k, v in list(obj.items())[:50]:
        if not isinstance(k, int) and not (isinstance(k, str) and k.isdigit()):
            keys_are_int = False
        if not isinstance(v, int) and not (isinstance(v, str) and str(v).isdigit()):
            vals_are_int = False

    if keys_are_int and not vals_are_int:
        inv = {}
        for k, v in obj.items():
            kk = int(k) if isinstance(k, str) and k.isdigit() else k
            inv[str(v)] = int(kk)
        obj = inv

    vocab = {}
    for k, v in obj.items():
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        vocab[str(k)] = int(v)

    if "<pad>" not in vocab:
        vocab["<pad>"] = 0
    if "<unk>" not in vocab:
        vocab["<unk>"] = 1
    return vocab

def _load_vocab_any(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing vocab: {path}")

    try:
        obj = _load_json(path)
        v = _coerce_vocab(obj)
        if v is not None:
            return v
    except Exception:
        pass

    obj = torch.load(path, map_location="cpu")
    v = _coerce_vocab(obj)
    if v is None:
        raise ValueError(f"Unsupported vocab format: {path}")
    return v

def _load_threshold():
    if not os.path.exists(THRESH_PATH):
        return 0.5
    obj = _load_json(THRESH_PATH)
    return float(obj.get("threshold", 0.5))

def _unwrap_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")

def _strip_prefixes(state):
    prefixes = ("module.", "model.", "net.")
    fixed = {}
    for k, v in state.items():
        kk = str(k)
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        fixed[kk] = v
    return fixed

def _infer_arch(state):
    tok_w = state["token_embedding.weight"]
    pos_w = state["position_embedding.weight"]
    vocab_size = int(tok_w.shape[0])
    embed_dim = int(tok_w.shape[1])
    max_len = int(pos_w.shape[0])

    layer_ids = set()
    for k in state.keys():
        if k.startswith("transformer_encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    ffn_key = None
    for k in state.keys():
        if k.endswith(".linear1.weight") and k.startswith("transformer_encoder.layers."):
            ffn_key = k
            break
    ffn_dim = int(state[ffn_key].shape[0]) if ffn_key is not None else embed_dim * 2

    for h in (8, 4, 16, 2, 1):
        if embed_dim % h == 0:
            num_heads = h
            break

    dropout = 0.1
    return vocab_size, embed_dim, num_heads, num_layers, ffn_dim, dropout, max_len

class SpamTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim, dropout, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classification_head = nn.Linear(embed_dim, 2)

    def forward(self, input_ids, attention_mask):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.classification_head(pooled)

def _encode(text, vocab, max_len):
    unk = vocab.get("<unk>", 1)
    toks = _tokenize(text)
    ids = [vocab.get(tok, unk) for tok in toks][:max_len]
    if len(ids) == 0:
        ids = [unk]
    known = sum(1 for tok in toks[:max_len] if tok in vocab)
    denom = max(1, min(len(toks), max_len))
    return ids, known / denom

def _read_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV missing header row")

        lower = {c.lower(): c for c in r.fieldnames}
        if "text" not in lower:
            raise ValueError(f"CSV must have a 'text' column. Found: {r.fieldnames}")
        text_col = lower["text"]
        path_col = lower.get("path", None)

        rows = []
        for row in r:
            text = row.get(text_col, "")
            tag = row.get(path_col, "") if path_col else ""
            rows.append((text, tag))
        return rows

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Missing vocab: {VOCAB_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = _load_vocab_any(VOCAB_PATH)
    threshold = _load_threshold()

    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state = _strip_prefixes(_unwrap_state(ckpt))

    vs, ed, h, layers, ffn, drop, max_len = _infer_arch(state)
    if vs != len(vocab):
        raise ValueError(f"Vocab size mismatch: checkpoint={vs} vocab={len(vocab)}")

    model = SpamTransformer(vs, ed, h, layers, ffn, drop, max_len).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = _read_csv(CSV_PATH)
    print(f"rows_loaded={len(rows)} csv={CSV_PATH}")

    with torch.inference_mode():
        for i, (text, tag) in enumerate(rows, start=1):
            ids, known_rate = _encode(text, vocab, max_len)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones((1, len(ids)), dtype=torch.bool, device=device)
            logits = model(input_ids, attention_mask)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].detach().cpu())
            label = "SPAM" if prob >= threshold else "NOT_SPAM"
            name = tag if tag else f"row={i}"
            if known_rate < 0.05:
                print(f"{label}\t{prob:.4f}\tknown={known_rate:.3f}\t{name}")
            else:
                print(f"{label}\t{prob:.4f}\t{name}")

if __name__ == "__main__":
    main()
