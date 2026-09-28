import torch
from tokenizer import get_tokenizer
from model import GPTModel
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer(TOKENIZER_NAME)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -torch.inf

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat([idx, next_id], dim=1)

    return idx


model = GPTModel(len(tokenizer)).to(device)
model.load_state_dict(torch.load("model_100.pth"))
model.eval()

context = input("Start: ")
idx = torch.tensor(tokenizer.encode(context)).unsqueeze(0).to(device)

out = generate(model, idx, 50, CONTEXT_LENGTH, temperature=0.5, top_k=50)

# print(tokenizer.decode(out[0].tolist()))