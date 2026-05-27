import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../extend")
from train_squad import RealQADataset

tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'mask_token': '[MASK]'})

print("Loading dataset...")
squad = load_dataset("squad", split="train[:50]")
dataset = RealQADataset(squad, tokenizer, block_size=256)

for i in range(15):
    input_ids, token_type_ids, masks, start, end = dataset[i]
    question = squad[i]['question']
    answer = squad[i]['answers']['text'][0] if len(squad[i]['answers']['text']) > 0 else "NO ANSWER"
    
    # Let's decode the tokens from start to end to see what the mapped answer is!
    mapped_tokens = input_ids[start:end+1]
    mapped_answer = tokenizer.decode(mapped_tokens)
    
    print(f"\nSample {i}:")
    print(f"  Question:      {question}")
    print(f"  Target Answer: {answer}")
    print(f"  Mapped Start:  {start.item()} | Mapped End: {end.item()}")
    print(f"  Mapped Text:   '{mapped_answer.strip()}'")
