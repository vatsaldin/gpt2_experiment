from transformers import GPT2TokenizerFast

print("Loading 'openai-community/gpt2'...")
tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
print("Vocab size:", len(tokenizer))
print("Encode 'Hello':", tokenizer.encode("Hello"))
print("Tokenize 'Hello':", tokenizer.tokenize("Hello"))
