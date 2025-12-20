import os
import re
import csv
from email import policy
from email.parser import BytesParser

EML_DIR = "emls"
OUT_CSV = os.path.join("emls", "eml_processed.csv")

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]", re.UNICODE)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

def list_emls(root):
    out = []
    for r, _, files in os.walk(root):
        for n in files:
            if n.lower().endswith(".eml"):
                out.append(os.path.join(r, n))
    out.sort()
    return out

def html_to_text(s):
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = TAG_RE.sub(" ", s)
    return s

def extract_subject_and_body(path):
    raw = None
    try:
        with open(path, "rb") as f:
            raw = f.read()
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        try:
            txt = raw.decode("utf-8", errors="replace") if raw is not None else ""
        except Exception:
            txt = ""
        return "", txt

    subject = str(msg.get("subject") or "").strip()

    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                try:
                    content = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        content = payload.decode(charset, errors="replace")
                    except Exception:
                        content = payload.decode("utf-8", errors="replace")
                parts.append((ctype, str(content)))
    else:
        try:
            parts.append((msg.get_content_type(), str(msg.get_content())))
        except Exception:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    parts.append((msg.get_content_type(), payload.decode(charset, errors="replace")))
                except Exception:
                    parts.append((msg.get_content_type(), payload.decode("utf-8", errors="replace")))

    body_plain = ""
    body_html = ""
    for ctype, content in parts:
        if ctype == "text/plain" and not body_plain.strip():
            body_plain = content
        if ctype == "text/html" and not body_html.strip():
            body_html = content

    body = body_plain if body_plain.strip() else html_to_text(body_html)

    if not body.strip() and raw is not None:
        try:
            body = raw.decode("utf-8", errors="replace")
        except Exception:
            body = ""

    return subject, body

def to_dataset_line(subject, body):
    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)
    s = subject.strip()
    b = body.strip()

    if s and b:
        combined = "Subject: " + s + "\n" + b
    elif s:
        combined = "Subject: " + s
    else:
        combined = b

    combined = combined.replace("\u200b", " ")
    combined = combined.replace("\r\n", "\n").replace("\r", "\n")
    combined = WS_RE.sub(" ", combined).strip()

    toks = TOKEN_RE.findall(combined)
    toks = [t.lower() for t in toks]
    text = " ".join(toks)

    text = text.replace("subject :", "subject:")
    if text.startswith("subject:"):
        text = "Subject:" + text[len("subject:"):]
    return text

def main():
    emls = list_emls(EML_DIR)
    if not emls:
        raise FileNotFoundError(f"No .eml files under {EML_DIR}")

    rows = []
    for p in emls:
        subject, body = extract_subject_and_body(p)
        line = to_dataset_line(subject, body)
        rows.append((line, p))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "path"])
        w.writerows(rows)

    print(f"wrote={OUT_CSV} rows={len(rows)}")

if __name__ == "__main__":
    main()
