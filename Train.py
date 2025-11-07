# -*- coding: utf-8 -*-
"""
MiniLLM Training Script
A self-contained script to train a small GPT-style language model from scratch.

Requirements:
    pip install torch requests

Usage:
    python train.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import requests
from pathlib import Path
import json

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

# ==================== Tokenizer ====================

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

# ==================== Model Components ====================

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

# ==================== Main Model ====================

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

# ==================== Dataset ====================

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

# ==================== Training Function ====================

def run_training(config=None):
    """
    Train a small LLM from scratch.
    
    Args:
        config (dict): Configuration dictionary. Uses defaults if None.
    
    Returns:
        tuple: (model, tokenizer)
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    device = torch.device(cfg['device'])
    cache_dir = Path(cfg['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    
    print("="*50)
    print("MiniLLM Training")
    print("="*50)
    print(f"Device: {device}")
    print(f"Cache directory: {cache_dir}")
    
    # Load data
    data_path = cache_dir / "tinystories.txt"
    tokenizer_path = cache_dir / "tokenizer.json"
    
    if not data_path.exists():
        print("Downloading TinyStories dataset...")
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-100M.txt"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text[:cfg['data_limit']])
        print(f"Data saved ({data_path.stat().st_size/1e6:.1f} MB)")
    
    # Prepare tokenizer
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if tokenizer_path.exists():
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        tokenizer = SimpleTokenizer(text)
        tokenizer.save(tokenizer_path)
    
    # Prepare datasets
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data, val_data = data[:n], data[n:]
    
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
                if True:  # sample_generation
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
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8)
    print("\n" + "="*50)
    print("FINAL GENERATION")
    print("="*50)
    print(tokenizer.decode(generated[0].tolist()))
