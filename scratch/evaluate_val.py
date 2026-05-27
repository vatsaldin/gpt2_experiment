import torch
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../extend")
from gpt_extend import HighContextQAModel, QAEncoderConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'mask_token': '[MASK]'})

config = QAEncoderConfig(
    vocab_size=len(tokenizer),
    block_size=256
)

model = HighContextQAModel(config).to(device)
checkpoint_path = "finetuned_qa.pt"
if not os.path.exists(checkpoint_path):
    print("Checkpoint not found!")
    sys.exit(1)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("Loading SQuAD validation split...")
squad = load_dataset("squad", split="validation")

print("\nEvaluating 5 random validation samples:")
for i in range(5):
    sample = squad[i * 20] # take spaced-out samples
    context = sample['context']
    question = sample['question']
    ground_truth = sample['answers']['text']
    
    encoding = tokenizer(
        question, 
        context,
        max_length=256,
        truncation="only_second", 
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Compute proper token_type_ids
    sequence_ids = encoding.sequence_ids(0)
    token_type_ids = torch.tensor([
        (1 if sid == 1 else 0) for sid in sequence_ids
    ], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        start_logits, end_logits, _ = model(
            idx=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
    start_logits = start_logits.squeeze(0).cpu()
    end_logits = end_logits.squeeze(0).cpu()
    offset_mapping = encoding["offset_mapping"].squeeze(0)
    
    best_score = -float("inf")
    best_span = (0, 0)
    
    for start in range(len(sequence_ids)):
        if sequence_ids[start] != 1:
            continue
        for end in range(start, min(start + 15, len(sequence_ids))):
            if sequence_ids[end] != 1:
                continue
            score = start_logits[start].item() + end_logits[end].item()
            if score > best_score:
                best_score = score
                best_span = (start, end)
                
    start_tok, end_tok = best_span
    
    if start_tok > 0:
        char_start = offset_mapping[start_tok][0].item()
        char_end = offset_mapping[end_tok][1].item()
        predicted_answer = context[char_start:char_end].strip()
    else:
        predicted_answer = "[No Answer Extracted]"
        
    print(f"\nSample {i+1}:")
    print(f"  Question:     {question}")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Predicted:    '{predicted_answer}' (span: {start_tok} to {end_tok}, score: {best_score:.4f})")
