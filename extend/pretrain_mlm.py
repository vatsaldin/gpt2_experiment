import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from gpt_extend import HighContextMLMModel, QAEncoderConfig
import warnings
warnings.filterwarnings("ignore")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # Add a specific mask token if not present
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    print("Loading Wikipedia dataset subset (wikitext-2-raw-v1)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]") # Use very small subset for testing

    print("Tokenizing dataset...")
    def tokenize_function(examples):
        # We process long texts and truncate to block size
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text']) > 10)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    config = QAEncoderConfig(
        vocab_size=len(tokenizer), # Important: update vocab size to match tokenizer
        block_size=256 # Reduced for local test execution speed
    )
    
    model = HighContextMLMModel(config).to(device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Basic MLM Data Collator logic
    def mlm_collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        
        labels = input_ids.clone()
        
        # 15% masking probability
        probability_matrix = torch.full(labels.shape, 0.15)
        
        # We do not mask pad tokens
        special_tokens_mask = (input_ids == tokenizer.pad_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Labels should be -100 for non-masked tokens to ignore in loss
        labels[~masked_indices] = -100 
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        
        return input_ids, attention_mask, labels

    dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True, collate_fn=mlm_collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Starting Pre-training Dry Run (max 20 steps)...")
    model.train()
    
    for step, (inputs, masks, labels) in enumerate(dataloader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        
        # Skip batch if no tokens are masked (causes NaN loss)
        if (labels != -100).sum() == 0:
            continue
            
        logits, loss = model(idx=inputs, attention_mask=masks, labels=labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Pre-train Step {step} | MLM Loss: {loss.item():.4f}")
            
        if step >= 20: # Cap for dry run
            break
            
    print("Pre-training Dry Run successful!")
    
    # Save base backbone weights
    torch.save(model.transformer.state_dict(), "pretrained_encoder.pt")
    print("Saved pretrained_encoder.pt")

if __name__ == "__main__":
    main()
