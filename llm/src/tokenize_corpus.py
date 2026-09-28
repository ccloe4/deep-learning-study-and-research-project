import numpy as np
from tokenizer import get_tokenizer
from pathlib import Path
from config import TOKENIZER_NAME

# 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_FILE = DATA_DIR / "corpus.txt"
TOKEN_FILE = DATA_DIR / "tokens.dat"

CHUNK_TOKEN_LIMIT = 100_000  # 한 번에 토큰화할 개수

tokenizer = get_tokenizer(TOKENIZER_NAME)

print("==== Starting memmap tokenization ====")

# 임시 list에 토큰 모으고, 일정량마다 memmap에 저장
batch_tokens = []
total_tokens = 0
first_save = True
memmap_arr = None

with open(CORPUS_FILE, "r", encoding="utf-8-sig") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        batch_tokens.extend(tokenizer.encode(line))

        if len(batch_tokens) >= CHUNK_TOKEN_LIMIT:
            chunk = np.array(batch_tokens, dtype=np.int32)
            if first_save:
                # memmap 초기 생성
                memmap_arr = np.memmap(TOKEN_FILE, dtype=np.int32, mode='w+', shape=(len(chunk),))
                memmap_arr[:] = chunk
                first_save = False
            else:
                # memmap append
                memmap_arr.flush()
                old_size = memmap_arr.shape[0]
                new_size = old_size + len(chunk)
                memmap_arr = np.memmap(TOKEN_FILE, dtype=np.int32, mode='r+', shape=(old_size,))
                # resize
                with open(TOKEN_FILE, 'ab') as f_append:
                    chunk.tofile(f_append)
                memmap_arr = np.memmap(TOKEN_FILE, dtype=np.int32, mode='r+', shape=(new_size,))
            batch_tokens = []
            total_tokens += len(chunk)

        if i % 100_000 == 0:
            print(f"Processed {i} lines, total tokens so far: {total_tokens}, current batch size: {len(batch_tokens)}")

# 마지막 남은 토큰
if batch_tokens:
    chunk = np.array(batch_tokens, dtype=np.int32)
    if first_save:
        memmap_arr = np.memmap(TOKEN_FILE, dtype=np.int32, mode='w+', shape=(len(chunk),))
        memmap_arr[:] = chunk
    else:
        memmap_arr.flush()
        with open(TOKEN_FILE, 'ab') as f_append:
            chunk.tofile(f_append)
    total_tokens += len(chunk)

print(f"==== Tokenization completed. Total tokens saved: {total_tokens} ====")