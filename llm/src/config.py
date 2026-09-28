# ======================
# Tokenizer
# ======================
TOKENIZER_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# ======================
# Model
# ======================
CONTEXT_LENGTH = 256  # Shortened context length (orig: 1024)
EMB_DIM = 512  # Embedding dimension 768 -> 무거움
NUM_HEADS = 8  # Number of attention heads 12 -> 무거움
NUM_LAYERS = 8  # Number of layers 12 -> 무거움
DROP_RATE = 0.1  # Dropout rate
QKV_BIAS = False  # Query-key-value bias

# ======================
# Dataset
# ======================
MAX_LENGTH = 64
STRIDE = 8

# ======================
# Training
# ======================
BATCH_SIZE = 64  # 128 -> OOM 가능
LR = 4e-4
WEIGHT_DECAY = 0.1

# LLM-style Control
TARGET_TOKENS = 40_000_000
LOG_INTERVAL = 100
SAVE_INTERVAL = 5_000_000
MAX_KEEP = 5  # 유지할 최신 checkpoint 수