from tokenizer.basic_tokenizer import SimpleTokenizerV2
from tokenizer.bpe_tokenizer import BytePairTokenizer
from experiments.compare_with_tiktoken import compare

import re

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":

    text = load_text("data/sample.txt")

    # Word-level tokenizer
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]

    vocab = sorted(list(set(tokens)))
    vocab.extend(["<|endoftext|>", "<|unk|>"])
    vocab_dict = {token: i for i, token in enumerate(vocab)}

    tokenizer = SimpleTokenizerV2(vocab_dict)

    encoded = tokenizer.encode(text)
    print("Word-level encoded:")
    print(encoded[:50])

    # BPE tokenizer
    bpe = BytePairTokenizer()
    bpe.train(text)

    encoded_bpe = bpe.encode(text)
    print("BPE encoded:")
    print(encoded_bpe[:50])

    print("Compression ratio:",
          len(text.encode("utf-8")) / len(encoded_bpe))

    # Compare with real GPT
    compare(text)