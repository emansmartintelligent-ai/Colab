# -*- coding: utf-8 -*-
"""
MiniLLM Training Script - Colab Optimized
Requirements: pip install torch datasets
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

# ==================== Configuration ====================

DEFAULT_CONFIG = {
    'n_embeddings': 384,
    'n_layers': 6,
    'n_heads': 6,
    'context_length': 256,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 5,
    'learning_rate': 3e-4,
    'eval_interval': 1000,
    'eval_iters': 200,
    'max_steps': None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cache_dir': './data',
    'model_save_path': 'minillm.pth',
    'data_limit': 10_000_000,
}

# ==================== Tokenizer & Model Components ====================

class SimpleTokenizer:
    """Character-level tokenizer for demonstration"""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = data['stoi']
        tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
        tokenizer.vocab_size = len(tokenizer.stoi)
        return tokenizer

class Head(nn.Module):
    """Single self-attention head"""
    def __init__(self, head_size, n_embeddings, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embeddings, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embeddings, context_length, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embeddings, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block with residual connections"""
    def __init__(self, n_embeddings, n_heads, context_length, dropout):
        super().__init__()
        head_size = n_embeddings // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embeddings, context_length, dropout)
        self.ffwd = FeedForward(n_embeddings, dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, n_embeddings, n_layers, n_heads, context_length, dropout):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding = nn.Embedding(context_length, n_embeddings)
        self.blocks = nn.Sequential(*[Block(n_embeddings, n_heads, context_length, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        return logits, F.cross_entropy(logits, targets)
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.context_length:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx

class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length
    
    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y

# ==================== Data Loading (Colab-Friendly) ====================

def prepare_data(cache_dir, data_limit):
    """Prepare data using Hugging Face datasets library (most reliable)"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    data_path = cache_dir / "tinystories.txt"
    tokenizer_path = cache_dir / "tokenizer.json"
    
    # Use this cell in Colab to install required packages:
    # !pip install datasets
    
    try:
        print("Loading TinyStories dataset via Hugging Face...")
        from datasets import load_dataset
        
        # Load dataset (cached automatically by HF)
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        # Convert to text and limit size
        print(f"Dataset loaded. Total examples: {len(dataset)}")
        
        # Take subset to avoid memory issues
        max_examples = data_limit // 500  # ~500 chars per example
        text = '\n'.join(dataset['text'][:max_examples])
        
        print(f"Using {len(text)/1e6:.1f}MB of text")
        
        # Save processed data
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
    except ImportError:
        print("⚠️  Hugging Face datasets not found. Install with: pip install datasets")
        print("⚠️  Falling back to synthetic data...")
        
        # Create synthetic stories
        synthetic_text = """Once upon a time, there was a little boy named Tim. Tim loved to play with his red ball.
One day, Tim lost his ball in the garden. He looked everywhere but could not find it.
Then, he saw his dog Max playing with the ball. Max was happy and wagged his tail.
Tim laughed and said, "Max, you found my ball!" They played together all day.

Lily was a girl who liked to draw. She drew a big house with a tree and a flower.
Her mom said, "Great job, Lily!" Lily smiled and put the picture on the wall.

The cat chased the mouse around the house. The mouse was fast and clever.
It hid in a small hole where the cat could not reach. The cat was sad and meowed.

Tom and his friend went to the park. They played on the swings and the slide.
They had a picnic with sandwiches and juice. The sun was shining bright.
""" * 200  # Repeat to create enough data
        
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(synthetic_text)
        
        print(f"Synthetic data created: {data_path}")
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Check your internet connection and Hugging Face access.")
        raise
    
    # Verify data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if len(text) < 1000:
        raise ValueError("Dataset appears to be corrupted or empty!")
    
    # Create tokenizer
    if tokenizer_path.exists():
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        tokenizer = SimpleTokenizer(text)
        tokenizer.save(tokenizer_path)
    
    # Prepare datasets
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data, val_data = data[:n], data[n:]
    
    return tokenizer, train_data, val_data

# ==================== Training Function ====================

def run_training(config=None):
    """Train a small LLM from scratch"""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    device = torch.device(cfg['device'])
    cache_dir = Path(cfg['cache_dir'])
    
    print("="*50)
    print("MiniLLM Training")
    print("="*50)
    print(f"Device: {device}")
    print(f"Cache directory: {cache_dir}")
    
    # Prepare data
    tokenizer, train_data, val_data = prepare_data(cfg['cache_dir'], cfg['data_limit'])
    
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # Data loaders
    train_loader = DataLoader(TextDataset(train_data, cfg['context_length']), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, cfg['context_length']), batch_size=cfg['batch_size'])
    
    # Model
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        n_embeddings=cfg['n_embeddings'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        context_length=cfg['context_length'],
        dropout=cfg['dropout']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')
    max_steps = cfg['max_steps'] or (cfg['epochs'] * len(train_loader))
    
    print(f"\nTraining for {max_steps:,} steps...")
    print("="*50)
    
    for epoch in range(cfg['epochs']):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward & backward
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            step += 1
            
            # Evaluation
            if step % cfg['eval_interval'] == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    train_loss = sum(model(xb.to(device), yb.to(device))[1].item() 
                                   for i, (xb, yb) in enumerate(train_loader) if i < cfg['eval_iters']) / cfg['eval_iters']
                    val_loss = sum(model(xb.to(device), yb.to(device))[1].item() 
                                 for i, (xb, yb) in enumerate(val_loader) if i < cfg['eval_iters']) / cfg['eval_iters']
                
                print(f"Step {step:6d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': {k: v for k, v in cfg.items() if k != 'device'},
                        'vocab_size': tokenizer.vocab_size,
                        'step': step
                    }, cfg['model_save_path'])
                    print(f"✓ Model saved (val_loss: {val_loss:.4f})")
                
                # Sample generation
                print("\n--- Generation Sample ---")
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=150, temperature=0.8)
                print(tokenizer.decode(generated[0].tolist()))
                print("--- End Sample ---\n")
                
                model.train()
            
            if step >= max_steps:
                break
        
        if step >= max_steps:
            break
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {cfg['model_save_path']}")
    
    return model, tokenizer

# ==================== Main Execution ====================

if __name__ == "__main__":
    # Run this in Colab first:
    # !pip install datasets
    
    # Quick demo configuration
    demo_config = {
        'n_embeddings': 256,  # Smaller for faster demo
        'n_layers': 4,
        'n_heads': 4,
        'batch_size': 32,
        'epochs': 2,
        'eval_interval': 500,
        'data_limit': 2_000_000,  # Use 2M chars for quick demo
    }
    
    model, tokenizer = run_training(demo_config)
    
    # Final generation example
    if model is not None:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=200, temperature=0.8)
        print("\n" + "="*50)
        print("FINAL GENERATION")
        print("="*50)
        print(tokenizer.decode(generated[0].tolist()))
