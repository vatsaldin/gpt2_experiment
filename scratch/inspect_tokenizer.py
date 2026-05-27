from transformers import GPT2TokenizerFast, GPT2Tokenizer

print("Fast:")
tokenizer_fast = GPT2TokenizerFast.from_pretrained('gpt2')
print("Class:", type(tokenizer_fast))
print("Vocab size:", len(tokenizer_fast))
print("Encode 'Hello':", tokenizer_fast.encode("Hello"))
print("Tokenize 'Hello':", tokenizer_fast.tokenize("Hello"))

print("\nSlow:")
tokenizer_slow = GPT2Tokenizer.from_pretrained('gpt2')
print("Class:", type(tokenizer_slow))
print("Vocab size:", len(tokenizer_slow))
print("Encode 'Hello':", tokenizer_slow.encode("Hello"))
print("Tokenize 'Hello':", tokenizer_slow.tokenize("Hello"))
