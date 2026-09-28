from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 1. Tokeninzer sanity check
text = "나는 오늘 학교에 갔다."

ids = tokenizer(text)["input_ids"]
decoded = tokenizer.decode(ids)

print(text)
print(decoded)

print(len(ids), tokenizer.tokenize(text))

# token efficiency 측정 0.3 ~ 0.6 적정

from pathlib import Path

text = Path("./data/clean_wikipedia.txt").read_text(encoding="utf-8")[:100_000]

ids = tokenizer(text)["input_ids"]

print("chars:", len(text))
print("tokens:", len(ids))
print("tokens/char:", len(ids) / len(text))

# 2. Dataset sanity check

from dataset import MyDataset

dataset = MyDataset(text, tokenizer, max_length=32, stride=4)
print(len(dataset))

x, y = dataset[0]

# 동일 shape 확인
print(x.shape)  # (32,)
print(y.shape)  # (32,)

# y = x shifted by 1 확인
print(x[:10])
print(y[:10])

# 3. Dataloader check

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=4)

x, y = next(iter(loader))

print(x.shape)  # (4, 32)
print(y.shape)

# 4. model forward check
from model import GPTModel
model = GPTModel(len(tokenizer))
logits = model(x)

# (batch, seq_len, vocab_size)
print(logits.shape)

# 5. loss 계산 check
import torch.nn.functional as F

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    y.view(-1)
)

print(loss.item())

# 6. single batch overfit check

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(300):
    logits = model(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        # loss → 0.1 이하까지 감소해야 통과
        print(i, loss.item())


# 7. generate check
idx = x[:1]  # 한 샘플

out = model(idx).argmax(-1)

print(tokenizer.decode(out[0].tolist()))