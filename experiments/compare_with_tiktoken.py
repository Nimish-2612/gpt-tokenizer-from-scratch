import tiktoken

def compare(text):
    print("Original text:")
    print(text)
    print()

    enc_gpt2 = tiktoken.get_encoding("gpt2")
    enc_gpt4 = tiktoken.get_encoding("cl100k_base")

    print("GPT-2 tokens:")
    print(enc_gpt2.encode(text))

    print("GPT-4 tokens:")
    print(enc_gpt4.encode(text))