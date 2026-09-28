from transformers import AutoTokenizer

def get_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
