import matplotlib.pyplot as plt
import sys
import re

train_log_file = sys.argv[1]
output_file = sys.argv[2]

steps = []
tokens_seen = []
loss = []

# 로그 파싱
pattern = re.compile(r"step (\d+) \| tokens (\d+) \| loss ([0-9.]+)")
with open(train_log_file, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            tokens_seen.append(int(m.group(2)))
            loss.append(float(m.group(3)))

# loss 곡선
plt.figure(figsize=(10,5))
plt.plot(tokens_seen, loss, marker='o')
plt.title("Training Loss vs Tokens Seen")
plt.xlabel("Tokens Seen")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(output_file)
plt.close()