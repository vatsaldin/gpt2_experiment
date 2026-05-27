import torch
from transformers import GPT2TokenizerFast
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../extend")
from gpt_extend import HighContextQAModel, QAEncoderConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
    "and facing it, is a copper statue of Christ with arms upraised. Next to the Main Building "
    "is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian "
    "place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the "
    "Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
)
question = "Lourdes France is where the Virgin Mary appeared to whom?"

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
token_type_ids = torch.zeros_like(input_ids).to(device)

with torch.no_grad():
    start_logits, end_logits, _ = model(
        idx=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
    )

start_logits = start_logits.squeeze(0).cpu()
end_logits = end_logits.squeeze(0).cpu()

sequence_ids = encoding.sequence_ids(0)
offset_mapping = encoding["offset_mapping"].squeeze(0)

best_score = -float("inf")
best_span = (0, 0)

for i in range(len(sequence_ids)):
    if sequence_ids[i] != 1: # Must be in context
        continue
    for j in range(i, min(i + 15, len(sequence_ids))):
        if sequence_ids[j] != 1:
            continue
        
        score = start_logits[i].item() + end_logits[j].item()
        if score > best_score:
            best_score = score
            best_span = (i, j)

start_token, end_token = best_span
print(f"Best span indices: start={start_token}, end={end_token}")
print(f"Best score: {best_score}")

for idx in [start_token, end_token]:
    token_str = tokenizer.decode([encoding["input_ids"][0][idx].item()])
    offset = offset_mapping[idx]
    print(f"Index {idx} | Token: '{token_str}' | Offset: {offset} | Char text: '{context[offset[0]:offset[1]]}'")
