import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import sys
sys.path.append("/Users/vatsaldesai/Projects/gpt2_experiment/extend")

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'mask_token': '[MASK]'})

print("Loading dataset...")
squad = load_dataset("squad", split="train[:1]")
item = squad[0]

question = item['question']
context = item['context']
answer_text = item['answers']['text'][0]
answer_char_start = item['answers']['answer_start'][0]

print("Question:", question)
print("Context:", context)
print("Answer:", answer_text, "Start char:", answer_char_start)

encoding = tokenizer(
    question, 
    context,
    max_length=256,
    truncation="only_second", 
    padding="max_length",
    return_offsets_mapping=True,
    return_tensors="pt"
)

sequence_ids = encoding.sequence_ids(0)
print("sequence_ids is None?", sequence_ids is None)
if sequence_ids is not None:
    print("sequence_ids content:", sequence_ids)
    print("Unique sequence IDs:", set(sequence_ids))
