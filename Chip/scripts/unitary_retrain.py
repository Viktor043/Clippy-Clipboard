#!/usr/bin/env python3
"""
Fine-tune TinyStories-1M with unitary-constrained attention weights.

Uses Cayley parametrization: U = (I - A)(I + A)^{-1} where A is skew-symmetric.
This guarantees U is always orthogonal/unitary, regardless of A's values.

If the model retains accuracy with unitary constraints, we only need
1 glass block per matrix (32 total) instead of SVD's 2 per matrix (64 total).
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

# Use CPU — model is tiny (1M params) and MPS has quirks with transformers embeddings
device = torch.device("cpu")
print(f"[UnitaryRetrain] Device: {device}")


class CayleyLinear(nn.Module):
    """
    Linear layer constrained to be orthogonal via Cayley parametrization.
    U = (I - A) @ (I + A)^{-1} where A is skew-symmetric (A = -A^T).
    """
    def __init__(self, n, bias_data=None):
        super().__init__()
        self.n = n
        # Skew-symmetric parameter: only upper triangle stored
        # A = triu(params) - triu(params)^T
        self.A_params = nn.Parameter(torch.zeros(n, n))
        self.bias = nn.Parameter(bias_data.clone()) if bias_data is not None else None
        # For compatibility
        self.in_features = n
        self.out_features = n

    def get_weight(self):
        """Compute orthogonal weight matrix from Cayley parametrization."""
        A = self.A_params - self.A_params.T  # skew-symmetric
        I = torch.eye(self.n, device=self.A_params.device)
        # U = (I - A) @ (I + A)^{-1}
        U = torch.linalg.solve(I + A, I - A)  # more numerically stable than inverse
        return U

    @property
    def weight(self):
        return self.get_weight()

    def forward(self, x):
        U = self.get_weight()
        out = x @ U.T
        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def from_weight(W, bias=None):
        """
        Initialize from an existing weight matrix W.
        Find A such that Cayley(A) ≈ nearest_unitary(W).
        """
        n = W.shape[0]
        assert W.shape == (n, n), f"Weight must be square, got {W.shape}"

        # Compute nearest unitary
        U_np, S, Vt = np.linalg.svd(W.detach().cpu().numpy(), full_matrices=True)
        U_orth = U_np @ Vt  # nearest orthogonal

        # Inverse Cayley: A = (I - U)(I + U)^{-1}
        I = np.eye(n)
        try:
            A = np.linalg.solve(I + U_orth, I - U_orth)
        except np.linalg.LinAlgError:
            A = np.zeros((n, n))

        # Make skew-symmetric
        A = (A - A.T) / 2

        layer = CayleyLinear(n, bias)
        layer.A_params.data = torch.tensor(A, dtype=torch.float32)

        # Verify
        U_recovered = layer.get_weight().detach().numpy()
        err = np.max(np.abs(U_recovered - U_orth))
        print(f"    Cayley init: max error from nearest unitary = {err:.2e}")

        return layer


def replace_attention_with_cayley(model):
    """Replace all attention projection weights with Cayley-constrained versions."""
    for layer_idx in range(model.config.num_layers):
        attn = model.transformer.h[layer_idx].attn.attention
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(attn, pname)
            W = proj.weight.data
            bias = proj.bias.data if proj.bias is not None else None
            print(f"  Layer {layer_idx} {pname}: ", end="")
            cayley = CayleyLinear.from_weight(W, bias)
            setattr(attn, pname, cayley)


class TinyStoriesDataset(Dataset):
    """Simple dataset from tokenizer output."""
    def __init__(self, texts, tokenizer, max_length=64):
        self.encodings = []
        for text in texts:
            enc = tokenizer(text, max_length=max_length, truncation=True,
                           padding="max_length", return_tensors="pt")
            self.encodings.append(enc["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# Load model and tokenizer
print("[UnitaryRetrain] Loading TinyStories-1M...")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Evaluation prompts
eval_prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
]


def get_model_device(model):
    """Get the device the model is on."""
    return next(model.parameters()).device

def compute_perplexity(model, tokenizer, prompts):
    model.eval()
    dev = get_model_device(model)
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * (inputs["input_ids"].shape[1] - 1)
            total_tokens += inputs["input_ids"].shape[1] - 1
    return np.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, max_new=30):
    model.eval()
    dev = get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=max_new,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# Baseline
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
baseline_text = generate_text(model, tokenizer, eval_prompts[0])
print(f"\n[Baseline] PPL={baseline_ppl:.4f}")
print(f"[Baseline] Text: {baseline_text[:100]}...")

# Replace with Cayley
print("\n[UnitaryRetrain] Replacing attention weights with Cayley parametrization...")
replace_attention_with_cayley(model)
model.to(device)

# Measure initial Cayley PPL (should match "nearest unitary" test — very high)
cayley_init_ppl = compute_perplexity(model, tokenizer, eval_prompts)
cayley_init_text = generate_text(model, tokenizer, eval_prompts[0])
print(f"\n[Cayley Init] PPL={cayley_init_ppl:.4f} (delta={((cayley_init_ppl-baseline_ppl)/baseline_ppl)*100:+.1f}%)")
print(f"[Cayley Init] Text: {cayley_init_text[:100]}...")

# Training data: simple stories
train_texts = [
    "Once upon a time there was a little girl named Lily who loved to play in the garden.",
    "The big dog ran across the field and jumped over the fence to chase the ball.",
    "Mom and dad took the children to the park where they played on the swings.",
    "A small bird sat on the branch and sang a beautiful song to the morning sun.",
    "The boy was happy because he got a new toy car for his birthday.",
    "She walked to school with her friend and they talked about their favorite books.",
    "The cat climbed up the tree and could not get down so dad helped it.",
    "It was a sunny day and the flowers in the garden were very pretty.",
    "The little rabbit hopped through the meadow looking for carrots to eat.",
    "Tom and his sister went to the beach and built a big sand castle.",
] * 100  # Repeat for more data

dataset = TinyStoriesDataset(train_texts, tokenizer, max_length=32)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Only train the Cayley parameters (A_params) and biases
cayley_params = []
other_params = []
for name, param in model.named_parameters():
    if "A_params" in name or "bias" in name:
        param.requires_grad = True
        cayley_params.append(param)
    else:
        param.requires_grad = False

print(f"\n[UnitaryRetrain] Trainable params: {sum(p.numel() for p in cayley_params):,} "
      f"(out of {sum(p.numel() for p in model.parameters()):,} total)")

optimizer = optim.AdamW(cayley_params, lr=1e-4, weight_decay=0.01)

# Fine-tune
print("\n[UnitaryRetrain] Fine-tuning with Cayley constraints...")
model.train()
for epoch in range(20):
    total_loss = 0
    n_batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cayley_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches

    if (epoch + 1) % 2 == 0 or epoch == 0:
        ppl = compute_perplexity(model, tokenizer, eval_prompts)
        text = generate_text(model, tokenizer, eval_prompts[0], max_new=40)
        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} PPL={ppl:.2f} "
              f"(delta={delta:+.1f}%) | {text[:80]}...")

# Final evaluation
print("\n[UnitaryRetrain] === FINAL RESULTS ===")
final_ppl = compute_perplexity(model, tokenizer, eval_prompts)
final_delta = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
print(f"  Baseline PPL: {baseline_ppl:.4f}")
print(f"  Cayley Init PPL: {cayley_init_ppl:.4f}")
print(f"  Final PPL: {final_ppl:.4f} (delta={final_delta:+.2f}%)")
for p in eval_prompts[:3]:
    text = generate_text(model, tokenizer, p, max_new=40)
    print(f"  '{p[:30]}...' → {text[:100]}...")

# Verify weights are still orthogonal
print("\n[UnitaryRetrain] Verifying orthogonality...")
max_orth_err = 0
for layer_idx in range(model.config.num_layers):
    attn = model.transformer.h[layer_idx].attn.attention
    for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        proj = getattr(attn, pname)
        if hasattr(proj, 'get_weight'):
            W = proj.get_weight().detach().cpu().numpy()
            orth_err = np.max(np.abs(W.T @ W - np.eye(64)))
            max_orth_err = max(max_orth_err, orth_err)
print(f"  Max orthogonality error across all matrices: {max_orth_err:.2e}")
print(f"  (Should be < 1e-5 for valid Cayley parametrization)")

print("\n[UnitaryRetrain] Done!")
