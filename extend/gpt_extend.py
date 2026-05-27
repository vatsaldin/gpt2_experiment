from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ARCHITECTURE CONFIGURATION
# ==========================================
@dataclass
class QAEncoderConfig:
    vocab_size: int = 1000     # Small vocab for demonstration/synthetic testing
    type_vocab_size: int = 2   # Segment embeddings (e.g. 0 for Question, 1 for Context)
    block_size: int = 2048     # High context window size
    n_layer: int = 4           # Balanced layer count to optimize parameter budget
    n_head: int = 6            # 6 heads * 64 dim = 384 embedding dimension
    n_embd: int = 384          # Matches your original embedding width
    dropout: float = 0.1       # Regularization for small QA datasets

# ==========================================
# 2. ZERO-PARAMETER ROTARY EMBEDDINGS (RoPE)
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device):
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)

def apply_rope(x, cos, sin):
    def rotate_half(tensor):
        x1, x2 = tensor[..., :tensor.shape[-1]//2], tensor[..., tensor.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)

# ==========================================
# 3. BIDIRECTIONAL ENCODER ATTENTION BLOCK
# ==========================================
class BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_rate = config.dropout

    def forward(self, x, cos, sin, attention_mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Apply RoPE transformations
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Apply scaled dot-product attention with optional mask
        # Expand mask for heads: (B, T) -> (B, 1, 1, T)
        if attention_mask is not None:
            # Mask should be boolean for PyTorch SDPA (True where we CAN attend, False for padding)
            # We assume incoming mask is 1 for valid, 0 for pad.
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        else:
            expanded_mask = None

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=expanded_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=False
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# ==========================================
# 4. CORE TRANSFORMER COMPONENTS
# ==========================================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class QAEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.attn(self.ln_1(x), cos, sin, attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# ==========================================
# 5. EXTRACTIVE QA MODEL TOPOLOGY
# ==========================================
class HighContextQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.type_vocab_size, config.n_embd), # Segment embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([QAEncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Adding bias back to the QA head is generally helpful for start/end baseline logits
        self.qa_outputs = nn.Linear(config.n_embd, 2, bias=True)
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, max_seq_len=config.block_size)
    
    def forward(self, idx, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        
        # Add segment embeddings if provided
        if token_type_ids is not None:
            token_type_embeds = self.transformer.wpe(token_type_ids)
            x = x + token_type_embeds
            
        x = self.transformer.drop(x)
        
        cos, sin = self.rope(T, idx.device)
        
        for block in self.transformer.h:
            x = block(x, cos, sin, attention_mask)
            
        x = self.transformer.ln_f(x)
        logits = self.qa_outputs(x) # Shape: (B, T, 2)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # Shape: (B, T)
        end_logits = end_logits.squeeze(-1)     # Shape: (B, T)

        loss = None
        if start_positions is not None and end_positions is not None:
            # If the answer is out of bounds (e.g. truncated), it will be set to ignored_index
            ignored_index = start_logits.size(-1) 
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            # CRITICAL FIX: Pass ignore_index so clamped values don't crash cross_entropy
            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            loss = (start_loss + end_loss) / 2
            
        return start_logits, end_logits, loss

# ==========================================
# 6. MASKED LANGUAGE MODEL TOPOLOGY (PRE-TRAINING)
# ==========================================
class HighContextMLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.type_vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([QAEncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.mlm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, max_seq_len=config.block_size)
    
    def forward(self, idx, token_type_ids=None, attention_mask=None, labels=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        
        if token_type_ids is not None:
            x = x + self.transformer.wpe(token_type_ids)
            
        x = self.transformer.drop(x)
        cos, sin = self.rope(T, idx.device)
        
        for block in self.transformer.h:
            x = block(x, cos, sin, attention_mask)
            
        x = self.transformer.ln_f(x)
        logits = self.mlm_head(x)
        
        loss = None
        if labels is not None:
            # Shift not needed for MLM (unlike causal LM)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            
        return logits, loss

# ==========================================
# 7. DATA SIMULATION (FOR EXECUTION TESTING)
# ==========================================
class SyntheticQADataset(Dataset):
    def __init__(self, num_samples, block_size, vocab_size):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate variable length sequence by random padding
        valid_len = torch.randint(self.block_size // 2, self.block_size, (1,)).item()
        
        # Create input ids with padding
        x = torch.zeros(self.block_size, dtype=torch.long)
        x[:valid_len] = torch.randint(1, self.vocab_size, (valid_len,)) # Pad index = 0
        
        # Create attention mask (1 for valid, 0 for padding)
        attention_mask = torch.zeros(self.block_size, dtype=torch.long)
        attention_mask[:valid_len] = 1
        
        # Simulate Question / Context segments
        question_len = torch.randint(10, 50, (1,)).item()
        token_type_ids = torch.zeros(self.block_size, dtype=torch.long)
        if question_len < valid_len:
            token_type_ids[question_len:valid_len] = 1 # Context gets type 1

        # Simulate start/end positions, including unanswerable chance
        if torch.rand(1).item() < 0.1: # 10% chance unanswerable
            start_pos, end_pos = 0, 0
        else:
            # Answer within the valid context (after question)
            if valid_len > question_len + 10:
                start_pos = torch.randint(question_len + 2, valid_len - 5, (1,)).item()
                end_pos = start_pos + torch.randint(1, 5, (1,)).item()
            else:
                start_pos, end_pos = 0, 0 # Fallback unanswerable
        
        return x, token_type_ids, attention_mask, torch.tensor(start_pos), torch.tensor(end_pos)

# ==========================================
# 7. TRAINING & EXECUTION HARNESS
# ==========================================
if __name__ == "__main__":
    # Device Assignment Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")

    # Set up configuration dimensions
    config = QAEncoderConfig()
    
    # Initialize the high-context model
    model = HighContextQAModel(config).to(device)
    print(f"Model successfully loaded. Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Generate synthetic validation loaders
    dataset = SyntheticQADataset(num_samples=256, block_size=config.block_size, vocab_size=config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Set up optimization trackers
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    print("\nStarting dry run execution for 3 epochs...")
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        
        for step, (x, token_type_ids, attention_mask, start_targets, end_targets) in enumerate(dataloader):
            x = x.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            start_targets = start_targets.to(device)
            end_targets = end_targets.to(device)

            # Forward pass execution
            start_logits, end_logits, loss = model(
                idx=x, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask, 
                start_positions=start_targets, 
                end_positions=end_targets
            )

            # Backward pass execution
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

            if step == 0 or (step + 1) % 10 == 0:
                print(f"Epoch {epoch} | Step {step+1:02d} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---")
                
    print("\nDry run and structural tests completed successfully.")