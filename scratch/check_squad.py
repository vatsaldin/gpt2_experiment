import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import sys
sys.path.append("/Users/vatsaldesai/Projects/gpt2_experiment/extend")
from train_squad import RealQADataset

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'mask_token': '[MASK]'})

print("Loading dataset...")
squad = load_dataset("squad", split="train[:100]")
dataset = RealQADataset(squad, tokenizer, block_size=256)

starts = []
ends = []
for i in range(100):
    input_ids, token_type_ids, masks, start, end = dataset[i]
    starts.append(start.item())
    ends.append(end.item())

print("First 20 starts:", starts[:20])
print("First 20 ends:", ends[:20])
print("Number of zeros in starts:", starts.count(0))
print("Number of non-zeros in starts:", len(starts) - starts.count(0))
