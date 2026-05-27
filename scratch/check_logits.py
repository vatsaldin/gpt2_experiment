import torch
from transformers import GPT2TokenizerFast
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../extend")
from gpt_extend import HighContextQAModel, QAEncoderConfig

# Device
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
model.load_state_dict(torch.load("finetuned_qa.pt", map_location=device))
model.eval()

context = (
    "Architecturally, the school Notre Dame has a Catholic character. Atop the Main Building's "
    "gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building "
    "and facing it, is a copper statue of Christ with arms upraised."
)
question = "What sits atop the gold dome?"

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

print("\n--- Start Logits top 10 ---")
top_start_vals, top_start_indices = torch.topk(start_logits, 10)
for val, idx in zip(top_start_vals, top_start_indices):
    token_str = tokenizer.decode([encoding["input_ids"][0][idx].item()])
    seq_id = encoding.sequence_ids(0)[idx]
    print(f"Index {idx:3d} | Logit: {val.item():6.4f} | Seq ID: {seq_id} | Token: '{token_str}'")

print("\n--- End Logits top 10 ---")
top_end_vals, top_end_indices = torch.topk(end_logits, 10)
for val, idx in zip(top_end_vals, top_end_indices):
    token_str = tokenizer.decode([encoding["input_ids"][0][idx].item()])
    seq_id = encoding.sequence_ids(0)[idx]
    print(f"Index {idx:3d} | Logit: {val.item():6.4f} | Seq ID: {seq_id} | Token: '{token_str}'")
