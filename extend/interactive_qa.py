import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os

# Append current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpt_extend import HighContextQAModel, QAEncoderConfig

def extract_answer(context, question, model, tokenizer, device, config):
    # Tokenize input context and question
    encoding = tokenizer(
        question, 
        context,
        max_length=config.block_size,
        truncation="only_second", 
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Compute proper token_type_ids (0 for question/special, 1 for context)
    sequence_ids = encoding.sequence_ids(0)
    token_type_ids = torch.tensor([
        (1 if sid == 1 else 0) for sid in sequence_ids
    ], dtype=torch.long).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        start_logits, end_logits, _ = model(
            idx=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
    start_logits = start_logits.squeeze(0).cpu() # Shape: (block_size,)
    end_logits = end_logits.squeeze(0).cpu()     # Shape: (block_size,)
    
    # We find the best valid start and end positions
    # Subject to: start <= end, end - start < 15, and tokens are part of the context (sequence_id == 1)
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
    
    if best_score == -float("inf") or start_token == 0:
        return "Unable to extract answer.", 0.0
        
    # Get answer boundaries in characters
    char_start = offset_mapping[start_token][0].item()
    char_end = offset_mapping[end_token][1].item()
    
    answer = context[char_start:char_end].strip()
    
    # Simple sigmoid pseudo-confidence based on logit sum
    confidence = torch.sigmoid(torch.tensor(best_score / 10.0)).item()
    
    return answer, confidence

def main():
    # Select Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    config = QAEncoderConfig(
        vocab_size=len(tokenizer),
        block_size=256
    )
    
    # Initialize QA Model and load fine-tuned weights
    model = HighContextQAModel(config).to(device)
    checkpoint_path = "finetuned_qa.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint file '{checkpoint_path}' not found in the workspace!")
        print("Please ensure you run 'python extend/train_squad.py --full' first.")
        return
        
    print(f"Loading fine-tuned QA weights from '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model successfully loaded!")
    
    # Pre-loaded sample context for interactive testing
    sample_context = (
        "Architecturally, the school Notre Dame has a Catholic character. Atop the Main Building's "
        "gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building "
        "and facing it, is a copper statue of Christ with arms upraised. Next to the Main Building "
        "is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian "
        "place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the "
        "Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
    )
    
    print("\n" + "="*60)
    print("      Welcome to the Custom GPT Bidirectional QA Model!")
    print("="*60)
    print("This interactive script lets you test your fine-tuned model.")
    print("You can use the default sample context or enter your own.")
    print("="*60)
    
    squad_val = None
    context = sample_context
    print(f"\n[Current Context]:\n{context}\n")
    
    while True:
        print("-" * 60)
        print("Options: ")
        print("  [1] Ask a question on the current context")
        print("  [2] Enter a completely custom context paragraph")
        print("  [3] Reset context to default Notre Dame paragraph")
        print("  [4] Load a random unseen SQuAD validation sample")
        print("  [5] Exit")
        choice = input("\nSelect option (1-5) [Default: 4]: ").strip()
        
        if choice == '1':
            question = input("\nEnter your question: ").strip()
            if not question:
                print("Question cannot be empty!")
                continue
                
            answer, conf = extract_answer(context, question, model, tokenizer, device, config)
            print("\n" + "*"*40)
            print(f"QUESTION:  {question}")
            print(f"ANSWER:    \033[1;32m{answer}\033[0m")
            print(f"CONFIDENCE: {conf:.2%}")
            print("*"*40 + "\n")
            
        elif choice == '2':
            new_ctx = input("\nPaste your custom context paragraph here:\n").strip()
            if len(new_ctx) < 20:
                print("Context paragraph is too short!")
            else:
                context = new_ctx
                print("\nContext successfully updated!")
                print(f"\n[Current Context]:\n{context}\n")
                
        elif choice == '3':
            context = sample_context
            print("\nContext reset to default Notre Dame paragraph.")
            print(f"\n[Current Context]:\n{context}\n")
            
        elif choice == '4' or choice == '':
            if squad_val is None:
                from datasets import load_dataset
                print("\nLoading SQuAD validation dataset (lazy loading)...")
                # Since squad is already downloaded, this will use local cache instantly
                squad_val = load_dataset("squad", split="validation")
                print("Validation dataset loaded successfully!")
                
            import random
            sample = random.choice(squad_val)
            context = sample['context']
            question = sample['question']
            
            # SQuAD validation samples can have multiple correct reference answers
            answers_list = list(set(sample['answers']['text']))
            ground_truth = " OR ".join([f"'{ans}'" for ans in answers_list])
            
            print("\n" + "="*60)
            print(f"[Random SQuAD Context]:\n{context}\n")
            print(f"QUESTION:      {question}")
            print(f"GROUND TRUTH:  {ground_truth}")
            print("="*60)
            
            print("\nProcessing extraction with model...")
            answer, conf = extract_answer(context, question, model, tokenizer, device, config)
            
            print("\n" + "*"*60)
            print(f"QUESTION:        {question}")
            print(f"MODEL PREDICTED: \033[1;32m{answer}\033[0m")
            print(f"GROUND TRUTH:    {ground_truth}")
            print(f"CONFIDENCE:      {conf:.2%}")
            print("*"*60 + "\n")
            
        elif choice == '5' or choice.lower() == 'exit':
            print("\nThank you for exploring the QA pipeline! Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
