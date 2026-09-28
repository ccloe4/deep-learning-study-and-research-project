import os
import glob
import torch
from torch.utils.data import DataLoader
from model import GPTModel
from pathlib import Path
from config import *
from dataset import MemmapTokenDataset, collate_fn 
from tokenizer import get_tokenizer

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 파일 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FILE_NAME = "tokens.dat"
FILE_DIR = DATA_DIR / FILE_NAME
CKPT_DIR = DATA_DIR / "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# Streaming Dataset
dataset = MemmapTokenDataset(FILE_DIR, MAX_LENGTH, STRIDE)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True
)


tokenizer = get_tokenizer(TOKENIZER_NAME)

model = GPTModel(len(tokenizer)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Resume from latest checkpoint if exists
checkpoint_files = sorted(glob.glob(str(CKPT_DIR / "model_tokens*_loss*.pth")),
                          key=os.path.getmtime)

tokens_seen = 0
global_step = 0

if checkpoint_files:
    last_ckpt = checkpoint_files[-1]
    print(f"Resuming from latest checkpoint: {last_ckpt}")
    ckpt = torch.load(last_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    tokens_seen = ckpt.get("tokens_seen", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"Resumed: tokens_seen={tokens_seen}, global_step={global_step}") 

print("******** START TRAINING ********")
model.train()

def save_checkpoint(model, tokens_seen, loss):
    ckpt_path = CKPT_DIR / f"model_tokens{tokens_seen}_loss{loss:.4f}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokens_seen": tokens_seen,
        "global_step": global_step
    }, ckpt_path)

    # 디스크 관리: 최신 MAX_KEEP개만 유지
    ckpts = sorted(glob.glob(str(CKPT_DIR / "model_tokens*_loss*.pth")), key=os.path.getmtime)
    while len(ckpts) > MAX_KEEP:
        os.remove(ckpts[0])
        ckpts.pop(0)
    print(f"Saved checkpoint: {ckpt_path}")

# Training loop
TOTAL_TARGET_TOKENS = TARGET_TOKENS + tokens_seen  # 누적 목표

while tokens_seen < TOTAL_TARGET_TOKENS:
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            y.flatten()
        )

        loss.backward()
        optimizer.step()

        # token 수 누적
        tokens_seen += x.numel()
        global_step += 1

        # 로그 출력
        if global_step % LOG_INTERVAL == 0:
            print(
                f"step {global_step} | "
                f"tokens {tokens_seen} | "
                f"loss {loss.item():.4f}"
            )

        if tokens_seen >= TOTAL_TARGET_TOKENS:
            print(f"Training finished at {tokens_seen} tokens")
            save_checkpoint(model, tokens_seen, loss.item())
            break

        elif tokens_seen % SAVE_INTERVAL < x.numel():
            save_checkpoint(model, tokens_seen, loss.item())