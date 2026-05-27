import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from gpt_extend import HighContextQAModel, QAEncoderConfig
import warnings
warnings.filterwarnings("ignore")

class RealQADataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size=256):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        context = item['context']
        
        answer_text = item['answers']['text'][0] if len(item['answers']['text']) > 0 else ""
        answer_char_start = item['answers']['answer_start'][0] if len(item['answers']['answer_start']) > 0 else -1
        
        encoding = self.tokenizer(
            question, 
            context,
            max_length=self.block_size,
            truncation="only_second", 
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        # GPT2 tokenizer doesn't return token_type_ids by default
        token_type_ids = torch.zeros_like(input_ids)
        
        offset_mapping = encoding["offset_mapping"].squeeze()
        
        start_token_pos = 0
        end_token_pos = 0
        
        if answer_char_start != -1:
            answer_char_end = answer_char_start + len(answer_text)
            # Find the tokens that correspond to these character offsets
            # The context is the second sequence
            sequence_ids = encoding.sequence_ids(0)
            
            # Start token
            idx_start = 0
            while idx_start < len(sequence_ids) and sequence_ids[idx_start] != 1:
                idx_start += 1
            
            # End token
            idx_end = len(sequence_ids) - 1
            while idx_end >= 0 and sequence_ids[idx_end] != 1:
                idx_end -= 1
                
            if idx_start < len(sequence_ids) and idx_end >= 0:
                # Map character to token
                context_start = offset_mapping[idx_start][0]
                context_end = offset_mapping[idx_end][1]
                
                if answer_char_start >= context_start and answer_char_end <= context_end:
                    start_idx = idx_start
                    while start_idx <= idx_end and offset_mapping[start_idx][0] <= answer_char_start:
                        start_idx += 1
                    start_token_pos = start_idx - 1
                    
                    end_idx = idx_end
                    while end_idx >= idx_start and offset_mapping[end_idx][1] >= answer_char_end:
                        end_idx -= 1
                    end_token_pos = end_idx + 1

        return input_ids, token_type_ids, attention_mask, torch.tensor(start_token_pos), torch.tensor(end_token_pos)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    print("Loading SQuAD dataset subset (first 200 samples)...")
    squad = load_dataset("squad", split="train[:200]")

    config = QAEncoderConfig(
        vocab_size=len(tokenizer),
        block_size=256 # Reduced for local test execution speed
    )

    model = HighContextQAModel(config).to(device)
    
    # Load pre-trained weights
    try:
        model.transformer.load_state_dict(torch.load("pretrained_encoder.pt"))
        print("Successfully loaded pre-trained transformer weights!")
    except FileNotFoundError:
        print("Pre-trained weights not found. Make sure to run pretrain_mlm.py first. Running from scratch.")

    dataset = RealQADataset(squad, tokenizer, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Starting SQuAD Fine-Tuning Dry Run (max 20 steps)...")
    model.train()
    
    for step, (inputs, token_type_ids, masks, start_targets, end_targets) in enumerate(dataloader):
        inputs, token_type_ids, masks = inputs.to(device), token_type_ids.to(device), masks.to(device)
        start_targets, end_targets = start_targets.to(device), end_targets.to(device)
        
        start_logits, end_logits, loss = model(
            idx=inputs, 
            token_type_ids=token_type_ids, 
            attention_mask=masks, 
            start_positions=start_targets, 
            end_positions=end_targets
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Fine-tune Step {step} | QA Loss: {loss.item():.4f}")
            
        if step >= 20:
            break
            
    print("SQuAD Fine-tuning Dry Run successful!")

if __name__ == "__main__":
    main()
