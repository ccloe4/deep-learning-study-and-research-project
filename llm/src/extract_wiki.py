from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"

def clean_wiki(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    text = re.sub(r"\[\[.*?\|", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = re.sub(r"[^\w\s가-힣.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r'style=".*?"', " ", text)
    text = re.sub(r'</?[^>]+>', " ", text)  # 남은 HTML
    return text


seen = set()

output_path = DATA_DIR / "clean_wikipedia.txt"

with open(output_path, "w", encoding="utf-8") as out:
    for file in EXTRACTED_DIR.rglob("*"):
        if not file.is_file():
            continue

        with open(file, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]

                # clean
                text = clean_wiki(text)
                
                # 강제 제거
                if "includeonly" in text:
                    continue
                
                # length filter
                if len(text) < 20:
                    continue
                if len(text) > 1000:
                    text = text[:1000]

                # dedup
                h = hash(text)
                if h in seen:
                    continue
                seen.add(h)

                if len(seen) > 1_000_000:
                    seen.clear()

                out.write(text + "\n")