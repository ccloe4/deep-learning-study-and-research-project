from datasets import load_dataset
import re

def clean_namuwiki(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[\[파일:.*?\]\]", " ", text)
    text = re.sub(r"\[\[.*?\|", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    text = re.sub(r"'''|''", "", text)
    text = re.sub(r"\|\|.*?\|\|", " ", text)
    text = re.sub(r"[^\w\s가-힣.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


dataset = load_dataset("heegyu/namuwiki-extracted", split="train", streaming=True)

seen = set()

with open("clean_namuwiki.txt", "w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"]

        # 1. clean
        text = clean_namuwiki(text)

        # 2. length filter
        if len(text) < 20:
            continue

        if len(text) > 1000:
            text = text[:1000]

        # 3. dedup (hash)
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)

        # optional: 메모리 보호
        if len(seen) > 1_000_000:
            seen.clear()

        f.write(text + "\n")