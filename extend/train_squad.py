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

        # Truncate question to a maximum of 120 tokens to avoid tokenizer truncation crashes
        question_tokens = self.tokenizer.encode(question)
        if len(question_tokens) > 120:
            question = self.tokenizer.decode(question_tokens[:120])
            
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
        # Compute proper token_type_ids (0 for question/special, 1 for context)
        sequence_ids = encoding.sequence_ids(0)
        token_type_ids = torch.tensor([
            (1 if sid == 1 else 0) for sid in sequence_ids
        ], dtype=torch.long)
        
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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune QA Model on SQuAD")
    parser.add_argument("--full", action="store_true", help="Fine-tune on the entire SQuAD train split")
    parser.add_argument("--steps", type=int, default=None, help="Force a specific max steps cap")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    # Hardware acceleration detection: CUDA -> MPS (for Apple Silicon Mac) -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    # If --full is provided, use the entire train split (~87k samples); otherwise, use 200 samples for quick dry run
    split_name = "train" if args.full else "train[:200]"
    print(f"Loading SQuAD dataset subset (split={split_name})...")
    squad = load_dataset("squad", split=split_name)

    config = QAEncoderConfig(
        vocab_size=len(tokenizer),
        block_size=256 # Reduced for local test execution speed
    )

    model = HighContextQAModel(config).to(device)
    
    # Load pre-trained weights from Stage 1
    try:
        model.transformer.load_state_dict(torch.load("pretrained_encoder.pt", map_location=device))
        print("Successfully loaded pre-trained transformer weights!")
    except FileNotFoundError:
        print("Pre-trained weights not found. Make sure to run pretrain_mlm.py first. Running from scratch.")

    dataset = RealQADataset(squad, tokenizer, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Calculate step constraints
    max_steps = args.steps if args.steps is not None else (3000 if args.full else 20)
    print_freq = 100 if args.full else 5

    print(f"Starting SQuAD Fine-Tuning ({'Full Run' if args.full else 'Dry Run'} | max {max_steps} steps)...")
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
        
        if step % print_freq == 0:
            print(f"Fine-tune Step {step} | QA Loss: {loss.item():.4f}")
            
        if step >= max_steps:
            break
            
    print("SQuAD Fine-tuning completed successfully!")
    
    # Save the fine-tuned model weights
    torch.save(model.state_dict(), "finetuned_qa.pt")
    print("Saved finetuned_qa.pt")

if __name__ == "__main__":
    main()
