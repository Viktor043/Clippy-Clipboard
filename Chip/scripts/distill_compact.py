#!/usr/bin/env python3
"""
Knowledge distillation: train a compact model that needs fewer glass blocks.

Target: d_model=64, num_layers=4 (half), d_ff=64 (4× smaller MLP)
This eliminates ALL 256×256 glass blocks and halves layer count.

Glass blocks needed:
  Per layer: 8 attn (64×64) + 4 MLP (all 64×64) = 12 blocks
  4 layers: 48 glass blocks (all 64×64) = 48 × 12.8mm = 61.4 cm
  vs original: 96 blocks (184 cm) — 50% fewer blocks, 67% less glass

Distillation approach:
  - Teacher: TinyStories-1M (8 layers, d_ff=256)
  - Student: TinyStories-compact (4 layers, d_ff=64)
  - Loss: KL divergence on logits + CE on labels
"""
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

device = torch.device("cpu")
print("[Distill] Device:", device)

# === Load teacher ===
print("[Distill] Loading teacher (TinyStories-1M)...")
teacher_name = "roneneldan/TinyStories-1M"
teacher = AutoModelForCausalLM.from_pretrained(teacher_name)
teacher.eval()
tokenizer = AutoTokenizer.from_pretrained(teacher_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Create student ===
print("[Distill] Creating compact student (4 layers, d_ff=64)...")
# Copy teacher config and modify
student_config = GPTNeoConfig(
    vocab_size=teacher.config.vocab_size,
    max_position_embeddings=teacher.config.max_position_embeddings,
    hidden_size=64,           # same as teacher
    num_layers=4,             # half of teacher (8 → 4)
    num_heads=16,             # same as teacher
    intermediate_size=64,     # 4× smaller MLP (256 → 64)
    attention_types=[[["global"], 4]],  # all global attention
    embed_dropout=0.0,
    attention_dropout=0.0,
    resid_dropout=0.0,
)
student = GPTNeoForCausalLM(student_config)

# Initialize student from teacher where possible
# Copy embeddings (same vocabulary, same hidden size)
student.transformer.wte.weight.data.copy_(teacher.transformer.wte.weight.data)
student.transformer.wpe.weight.data.copy_(
    teacher.transformer.wpe.weight.data[:student_config.max_position_embeddings])

# Copy first 4 layers' attention weights (same structure)
for i in range(student_config.num_layers):
    # Map student layer i to teacher layer (i * 2) to spread across teacher depth
    teacher_idx = i * 2
    s_attn = student.transformer.h[i].attn.attention
    t_attn = teacher.transformer.h[teacher_idx].attn.attention
    for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        s_proj = getattr(s_attn, pname)
        t_proj = getattr(t_attn, pname)
        s_proj.weight.data.copy_(t_proj.weight.data)
        if s_proj.bias is not None and t_proj.bias is not None:
            s_proj.bias.data.copy_(t_proj.bias.data)

    # MLP: student has d_ff=64, teacher has d_ff=256
    # Initialize with truncated teacher weights (first 64 rows/cols)
    s_mlp = student.transformer.h[i].mlp
    t_mlp = teacher.transformer.h[teacher_idx].mlp
    # c_fc: teacher (256, 64) → student (64, 64)
    s_mlp.c_fc.weight.data.copy_(t_mlp.c_fc.weight.data[:64, :])
    if s_mlp.c_fc.bias is not None:
        s_mlp.c_fc.bias.data.copy_(t_mlp.c_fc.bias.data[:64])
    # c_proj: teacher (64, 256) → student (64, 64)
    s_mlp.c_proj.weight.data.copy_(t_mlp.c_proj.weight.data[:, :64])
    if s_mlp.c_proj.bias is not None:
        s_mlp.c_proj.bias.data.copy_(t_mlp.c_proj.bias.data)

    # LayerNorm
    student.transformer.h[i].ln_1.weight.data.copy_(teacher.transformer.h[teacher_idx].ln_1.weight.data)
    student.transformer.h[i].ln_1.bias.data.copy_(teacher.transformer.h[teacher_idx].ln_1.bias.data)
    student.transformer.h[i].ln_2.weight.data.copy_(teacher.transformer.h[teacher_idx].ln_2.weight.data)
    student.transformer.h[i].ln_2.bias.data.copy_(teacher.transformer.h[teacher_idx].ln_2.bias.data)

# Copy final LN and LM head
student.transformer.ln_f.weight.data.copy_(teacher.transformer.ln_f.weight.data)
student.transformer.ln_f.bias.data.copy_(teacher.transformer.ln_f.bias.data)
student.lm_head.weight.data.copy_(teacher.lm_head.weight.data)

s_params = sum(p.numel() for p in student.parameters())
t_params = sum(p.numel() for p in teacher.parameters())
print(f"  Teacher params: {t_params:,}")
print(f"  Student params: {s_params:,} ({s_params/t_params*100:.1f}% of teacher)")

# === Training data ===
# Use simple prompts for distillation
train_prompts = [
    "Once upon a time, there was a little girl named",
    "The boy went to the park and saw a big",
    "She loved to play with her friends in the",
    "One day, a cat and a dog met in the",
    "The sun was bright and the sky was",
    "He was very happy because he found a new",
    "They went to the store to buy some",
    "The little bird flew over the tall",
    "Mom said it was time to go to",
    "She put on her red dress and went to the",
    "The fish swam in the big blue",
    "He liked to read books about magic and",
    "The flowers in the garden were very",
    "She was sad because she lost her favorite",
    "The rabbit hopped through the green",
    "They played together until the sun went",
    "He ate a yummy cake with chocolate",
    "The dog wagged his tail because he was",
    "She sang a pretty song about the",
    "The wind blew and the leaves started to",
]

eval_prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
]


def compute_perplexity(model, tokenizer, prompts):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * (inputs["input_ids"].shape[1] - 1)
            total_tokens += inputs["input_ids"].shape[1] - 1
    return np.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, max_new=40):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=max_new,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# === Pre-distillation eval ===
teacher_ppl = compute_perplexity(teacher, tokenizer, eval_prompts)
student_ppl = compute_perplexity(student, tokenizer, eval_prompts)
print(f"\n[Pre-distillation]")
print(f"  Teacher PPL: {teacher_ppl:.4f}")
print(f"  Student PPL: {student_ppl:.4f}")
print(f"  Teacher: {generate_text(teacher, tokenizer, eval_prompts[0])[:80]}...")
print(f"  Student: {generate_text(student, tokenizer, eval_prompts[0])[:80]}...")

# === Distillation training ===
print("\n[Distill] Starting distillation training...")
teacher.eval()
student.train()
student.to(device)
teacher.to(device)

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
temperature = 2.0
alpha = 0.5  # blend of distillation loss (KL) and hard loss (CE)

n_epochs = 50
best_ppl = float('inf')

for epoch in range(1, n_epochs + 1):
    total_loss = 0.0
    student.train()

    for prompt in train_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids)
            teacher_logits = teacher_outputs.logits

        student_outputs = student(input_ids=input_ids, labels=input_ids)
        student_logits = student_outputs.logits
        hard_loss = student_outputs.loss

        # KL divergence on soft targets
        soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher.exp(), reduction='batchmean') * (temperature ** 2)

        loss = alpha * kl_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_prompts)

    if epoch % 5 == 0 or epoch == 1:
        student_ppl = compute_perplexity(student, tokenizer, eval_prompts)
        text = generate_text(student, tokenizer, eval_prompts[0])
        print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} PPL={student_ppl:.2f} | {text[:70]}...")
        if student_ppl < best_ppl:
            best_ppl = student_ppl

print(f"\n[Distill] === FINAL RESULTS ===")
final_ppl = compute_perplexity(student, tokenizer, eval_prompts)
print(f"  Teacher PPL: {teacher_ppl:.4f}")
print(f"  Student PPL: {final_ppl:.4f} (best: {best_ppl:.4f})")
delta = ((final_ppl - teacher_ppl) / teacher_ppl) * 100
print(f"  PPL delta:   {delta:+.1f}%")

print(f"\n[Distill] === TEXT COMPARISON ===")
for p in eval_prompts:
    t_text = generate_text(teacher, tokenizer, p)
    s_text = generate_text(student, tokenizer, p)
    print(f"  '{p[:35]}...'")
    print(f"    Teacher: {t_text[:70]}...")
    print(f"    Student: {s_text[:70]}...")

print(f"\n[Distill] === GLASS ARCHITECTURE ===")
print(f"  Teacher: 96 glass blocks (80× 64×64 + 16× 256×256), 184 cm")
print(f"  Student: {4 * 12} glass blocks (ALL 64×64), {4 * 12 * 12.8 / 10:.1f} cm")
print(f"  Reduction: {(1 - 48/96)*100:.0f}% fewer blocks, {(1 - 48*12.8/1843)*100:.0f}% less glass")
print(f"  All blocks are 64×64 (proven at 100dB)")
print(f"  No 256×256 blocks needed!")
print(f"  Student params: {s_params:,} ({s_params/t_params*100:.1f}% of teacher)")
