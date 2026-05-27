from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
question = "Who are you?"
context = "I am a helpful assistant."

encoding = tokenizer(
    question, 
    context,
    return_offsets_mapping=True
)

print("sequence_ids:", encoding.sequence_ids(0))
print("offset_mapping:", encoding["offset_mapping"])

# Print tokens alongside sequence_ids and offset_mapping
for i, (ids, seq_id, offset) in enumerate(zip(encoding["input_ids"], encoding.sequence_ids(0), encoding["offset_mapping"])):
    print(f"Token {i}: id={ids}, decoded='{tokenizer.decode([ids])}', sequence_id={seq_id}, offset={offset}")
