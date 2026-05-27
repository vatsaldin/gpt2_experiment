from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
question = "What is your name?"
context = "My name is John."

res = tokenizer(question, context)
print("Keys:", res.keys())
print("input_ids:", res["input_ids"])
print("Decoded:", tokenizer.decode(res["input_ids"]))
