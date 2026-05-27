import torch
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

question = "What is your name?"
context = "My name is John. I live in New York."

# Without return_tensors
encoding = tokenizer(
    question, 
    context,
    max_length=50,
    truncation="only_second", 
    padding="max_length",
    return_offsets_mapping=True
)

print("Without return_tensors:")
print("sequence_ids:", encoding.sequence_ids(0))
print("Unique sequence_ids:", set(encoding.sequence_ids(0)))

# Let's inspect other keys
print("Keys:", encoding.keys())
print("Offsets mapping length:", len(encoding["offset_mapping"]))
print("Offsets mapping:", encoding["offset_mapping"][:15])

# Let's see if we can find where the second sequence starts by finding where offsets reset!
# Usually, the first sequence offsets increase, then reset to 0 or continue?
# Let's print input ids and their decoded text
for i, (ids, offsets) in enumerate(zip(encoding["input_ids"], encoding["offset_mapping"])):
    if ids == tokenizer.pad_token_id:
        continue
    decoded = tokenizer.decode([ids])
    print(f"Token {i}: id={ids}, offsets={offsets}, decoded='{decoded}'")
