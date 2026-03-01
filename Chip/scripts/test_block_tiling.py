#!/usr/bin/env python3
"""
Test all-64×64 block tiling approach for MLP.

Instead of 256×256 glass blocks, split each rectangular MLP into four 64×64 blocks.

W_fc (256×64) = [W1; W2; W3; W4] where each Wi is 64×64
  → 4 parallel pipelines: glass(Vi^T) → E(Si) → glass(Ui) → concat outputs

W_proj (64×256) = [W1 | W2 | W3 | W4] where each Wi is 64×64
  → split input into 4 chunks, 4 parallel pipelines, SUM outputs electronically

This increases glass count but keeps ALL blocks at 64×64 (proven size).
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[BlockTile] Loading TinyStories-1M...")
model_name = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class TiledSVDLinear(nn.Module):
    """Split rectangular matrix into 64×64 tiles, each with its own SVD."""
    def __init__(self, W, bias=None, tile_size=64):
        super().__init__()
        self.m, self.n = W.shape
        self.tile_size = tile_size

        if self.m > self.n:
            # Tall matrix (e.g., 256×64): split rows into tiles
            self.mode = "tall"
            n_tiles = self.m // tile_size
            assert self.m % tile_size == 0, f"m={self.m} not divisible by {tile_size}"
            assert self.n <= tile_size, f"n={self.n} > tile_size={tile_size}"

            self.tiles = nn.ModuleList()
            for i in range(n_tiles):
                Wi = W[i*tile_size:(i+1)*tile_size, :]  # tile_size × n
                # Pad to square if needed
                if Wi.shape[0] != Wi.shape[1]:
                    Wi_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                    Wi_padded[:Wi.shape[0], :Wi.shape[1]] = Wi
                    Wi = Wi_padded
                U, S, Vt = np.linalg.svd(Wi, full_matrices=True)
                tile = nn.Module()
                tile.U = nn.Parameter(torch.tensor(U, dtype=torch.float32), requires_grad=False)
                tile.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
                tile.Vt = nn.Parameter(torch.tensor(Vt, dtype=torch.float32), requires_grad=False)
                tile.orig_rows = min(tile_size, self.m - i*tile_size)
                self.tiles.append(tile)

        elif self.n > self.m:
            # Wide matrix (e.g., 64×256): split columns into tiles
            self.mode = "wide"
            n_tiles = self.n // tile_size
            assert self.n % tile_size == 0, f"n={self.n} not divisible by {tile_size}"
            assert self.m <= tile_size, f"m={self.m} > tile_size={tile_size}"

            self.tiles = nn.ModuleList()
            for i in range(n_tiles):
                Wi = W[:, i*tile_size:(i+1)*tile_size]  # m × tile_size
                # Pad to square if needed
                if Wi.shape[0] != Wi.shape[1]:
                    Wi_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                    Wi_padded[:Wi.shape[0], :Wi.shape[1]] = Wi
                    Wi = Wi_padded
                U, S, Vt = np.linalg.svd(Wi, full_matrices=True)
                tile = nn.Module()
                tile.U = nn.Parameter(torch.tensor(U, dtype=torch.float32), requires_grad=False)
                tile.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
                tile.Vt = nn.Parameter(torch.tensor(Vt, dtype=torch.float32), requires_grad=False)
                tile.orig_cols = min(tile_size, self.n - i*tile_size)
                self.tiles.append(tile)
        else:
            # Square: just normal SVD
            self.mode = "square"
            U, S, Vt = np.linalg.svd(W, full_matrices=True)
            tile = nn.Module()
            tile.U = nn.Parameter(torch.tensor(U, dtype=torch.float32), requires_grad=False)
            tile.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
            tile.Vt = nn.Parameter(torch.tensor(Vt, dtype=torch.float32), requires_grad=False)
            self.tiles = nn.ModuleList([tile])

        if bias is not None:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        if self.mode == "tall":
            # W_fc (256×64): split rows, same input to all tiles, concat outputs
            # Each tile: x(64) → Vt(64×64) → S(64) → U(64×64) → out(64)
            outputs = []
            for tile in self.tiles:
                z = x @ tile.Vt.T
                z = z * tile.S
                z = z @ tile.U.T
                outputs.append(z[:, :tile.orig_rows] if hasattr(tile, 'orig_rows') else z)
            # Actually for padded tiles, we need to take only the valid output rows
            # But since we padded W with zeros, the extra rows of U produce zeros anyway
            out = torch.cat(outputs, dim=-1)
            # Trim to actual output size
            out = out[..., :self.m]

        elif self.mode == "wide":
            # W_proj (64×256): split columns, split input, sum outputs
            out = torch.zeros(*x.shape[:-1], self.m, device=x.device)
            ts = self.tile_size
            for i, tile in enumerate(self.tiles):
                xi = x[..., i*ts:(i+1)*ts]  # (batch, tile_size)
                # Pad input to tile_size if needed
                if xi.shape[-1] < ts:
                    pad = torch.zeros(*xi.shape[:-1], ts - xi.shape[-1], device=x.device)
                    xi = torch.cat([xi, pad], dim=-1)
                z = xi @ tile.Vt.T
                z = z * tile.S
                z = z @ tile.U.T
                # Only take valid output rows
                out = out + z[..., :self.m]
            # Note: out already summed
        else:
            tile = self.tiles[0]
            z = x @ tile.Vt.T
            z = z * tile.S
            out = z @ tile.U.T

        if self.bias is not None:
            out = out + self.bias
        return out


def compute_perplexity(model, tokenizer, prompts):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * (inputs["input_ids"].shape[1] - 1)
            total_tokens += inputs["input_ids"].shape[1] - 1
    return np.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, max_new=40):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=max_new,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


eval_prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
]

# Baseline
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
baseline_text = generate_text(model, tokenizer, eval_prompts[0])
print(f"\n[Baseline] PPL={baseline_ppl:.4f}")
print(f"[Baseline] Text: {baseline_text[:80]}...")

# Replace MLP with tiled SVD
print("\n[BlockTile] Replacing MLP with tiled 64×64 SVD blocks...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

for layer_idx in range(model.config.num_layers):
    mlp = model.transformer.h[layer_idx].mlp

    W_fc = mlp.c_fc.weight.data.numpy()
    bias_fc = mlp.c_fc.bias.data.numpy() if mlp.c_fc.bias is not None else None
    mlp.c_fc = TiledSVDLinear(W_fc, bias_fc, tile_size=64)

    W_proj = mlp.c_proj.weight.data.numpy()
    bias_proj = mlp.c_proj.bias.data.numpy() if mlp.c_proj.bias is not None else None
    mlp.c_proj = TiledSVDLinear(W_proj, bias_proj, tile_size=64)

    # Verify per-layer
    x_test = torch.randn(1, 64)
    orig_fc = torch.tensor(W_fc, dtype=torch.float32)
    orig_out = x_test @ orig_fc.T + (torch.tensor(bias_fc) if bias_fc is not None else 0)
    tiled_out = mlp.c_fc(x_test)
    fc_err = (orig_out - tiled_out).abs().max().item()

    x_test2 = torch.randn(1, 256)
    orig_proj = torch.tensor(W_proj, dtype=torch.float32)
    orig_out2 = x_test2 @ orig_proj.T + (torch.tensor(bias_proj) if bias_proj is not None else 0)
    tiled_out2 = mlp.c_proj(x_test2)
    proj_err = (orig_out2 - tiled_out2).abs().max().item()

    print(f"  Layer {layer_idx}: fc_err={fc_err:.2e}, proj_err={proj_err:.2e}")

tiled_ppl = compute_perplexity(model, tokenizer, eval_prompts)
delta = ((tiled_ppl - baseline_ppl) / baseline_ppl) * 100

print(f"\n[BlockTile] === RESULTS ===")
print(f"  Baseline PPL: {baseline_ppl:.4f}")
print(f"  Tiled PPL:    {tiled_ppl:.4f} (delta={delta:+.4f}%)")

print(f"\n[BlockTile] === TEXT COMPARISON ===")
all_match = True
for p in eval_prompts:
    tiled_text = generate_text(model, tokenizer, p)
    baseline_t = generate_text(
        AutoModelForCausalLM.from_pretrained(model_name).eval(), tokenizer, p)
    match = "MATCH" if tiled_text == baseline_t else "DIFFER"
    if match != "MATCH":
        all_match = False
    print(f"  '{p[:35]}...' → {match}")
    if match == "DIFFER":
        print(f"    Baseline: {baseline_t[:70]}...")
        print(f"    Tiled:    {tiled_text[:70]}...")

print(f"\n[BlockTile] === ARCHITECTURE COMPARISON ===")
print(f"  Original SVD approach:  96 glass blocks (80× 64×64 + 16× 256×256), 184 cm")
print(f"  Block tiling approach: 128 glass blocks (ALL 64×64), {128 * 12.8 / 10:.1f} cm")
print(f"  Reduction: NO 256×256 blocks needed!")
print(f"  Trade-off: +32 glass blocks, but all are proven size (100dB)")
print(f"  PPL change: {delta:+.4f}%")
print(f"  Text match: {'ALL MATCH' if all_match else 'SOME DIFFER'}")
