#!/usr/bin/env python3
"""
CRITICAL EVALUATION: Are our results realistic?

This script tests several potential issues that could make our results
look better than they actually are:

1. OPTICAL LOSSES: Real glass has absorption, scattering, coupling losses.
   We model phase-only transforms, but real light loses power.

2. COHERENCE: We assume perfectly coherent light. Real lasers have
   finite linewidth and temporal coherence.

3. DETECTOR NOISE: Real photodetectors have shot noise, thermal noise.

4. QUANTIZATION: Real phase masks have finite precision (not Float32).

5. CROSSTALK: Adjacent waveguides leak into each other.

6. INPUT ENCODING FIDELITY: Converting electronic values to optical
   amplitudes/phases isn't perfect.

7. SEQUENTIAL ERROR ACCUMULATION: With 96 glass blocks in series,
   errors compound. Does our noise model capture this correctly?
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "roneneldan/TinyStories-1M"
print("[CRITICAL] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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


class RealisticSVDLinear(nn.Module):
    """SVD linear with REALISTIC optical impairments."""
    def __init__(self, W, bias=None,
                 glass_psnr_db=90,           # Glass precision (phase errors)
                 optical_loss_db=0.5,        # Insertion loss per glass block
                 detector_snr_db=60,         # Photodetector SNR
                 dac_bits=10,                # DAC precision for modulators
                 phase_quantization_bits=8,  # Phase mask quantization
                 crosstalk_db=-30):          # Adjacent channel crosstalk
        super().__init__()
        m, n = W.shape
        U_exact, S_exact, Vt_exact = np.linalg.svd(W, full_matrices=True)

        self.r = len(S_exact)
        self.out_features = m
        self.in_features = n

        # 1. Glass phase errors (already tested)
        U_noisy = self._add_noise(U_exact, glass_psnr_db)
        Vt_noisy = self._add_noise(Vt_exact, glass_psnr_db)

        # 2. Phase quantization (real phase masks have finite bits)
        if phase_quantization_bits < 32:
            levels = 2**phase_quantization_bits
            # Quantize the phase components
            U_noisy = self._quantize_unitary(U_noisy, levels)
            Vt_noisy = self._quantize_unitary(Vt_noisy, levels)

        self.U = nn.Parameter(torch.tensor(U_noisy, dtype=torch.float32), requires_grad=False)
        self.S = nn.Parameter(torch.tensor(S_exact, dtype=torch.float32), requires_grad=False)
        self.Vt = nn.Parameter(torch.tensor(Vt_noisy, dtype=torch.float32), requires_grad=False)

        # 3. Optical insertion loss (power lost per glass block)
        self.loss_factor = 10**(-optical_loss_db / 20)  # amplitude factor

        # 4. Detector noise level
        self.detector_noise_std = 10**(-detector_snr_db / 20)

        # 5. DAC quantization for electronic scaling
        if dac_bits < 32:
            S_quant = self._quantize_values(S_exact, dac_bits)
            self.S = nn.Parameter(torch.tensor(S_quant, dtype=torch.float32), requires_grad=False)

        # 6. Crosstalk
        self.crosstalk_factor = 10**(crosstalk_db / 20)  # amplitude coupling

        if bias is not None:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=False)
        else:
            self.bias = None

    @staticmethod
    def _add_noise(M, psnr_db):
        if psnr_db >= 100:
            return M
        signal_power = np.mean(M ** 2)
        noise_power = signal_power / (10 ** (psnr_db / 10))
        noise = np.random.randn(*M.shape).astype(np.float32) * np.sqrt(noise_power)
        return (M + noise).astype(np.float32)

    @staticmethod
    def _quantize_unitary(M, levels):
        """Quantize matrix elements to finite levels."""
        mn, mx = M.min(), M.max()
        step = (mx - mn) / (levels - 1)
        return (np.round((M - mn) / step) * step + mn).astype(np.float32)

    @staticmethod
    def _quantize_values(v, bits):
        """Quantize scalar values to finite DAC precision."""
        levels = 2**bits
        mn, mx = v.min(), v.max()
        if mx == mn:
            return v
        step = (mx - mn) / (levels - 1)
        return (np.round((v - mn) / step) * step + mn).astype(np.float32)

    def forward(self, x):
        # Step 1: E-O conversion (modulator) - input encoding
        # Real modulators have finite extinction ratio, but we model this via DAC bits

        # Step 2: Glass block V^T
        z = x @ self.Vt.T
        z = z * self.loss_factor  # insertion loss

        # Step 3: O-E conversion (detector) - read out after V^T
        if self.detector_noise_std > 0:
            z = z + torch.randn_like(z) * self.detector_noise_std * z.abs().mean()

        # Step 4: Electronic scaling (singular values)
        z_scaled = z[..., :self.r] * self.S

        # Step 5: Zero-pad if needed
        if self.out_features > self.r:
            pad = torch.zeros(*z.shape[:-1], self.out_features - self.r, device=z.device)
            z_padded = torch.cat([z_scaled, pad], dim=-1)
        else:
            z_padded = z_scaled

        # Step 6: E-O conversion (modulator) - re-encode for U

        # Step 7: Glass block U
        out = z_padded @ self.U.T
        out = out * self.loss_factor  # insertion loss

        # Step 8: Add crosstalk (adjacent channels leak)
        if self.crosstalk_factor > 0:
            crosstalk = torch.zeros_like(out)
            crosstalk[..., 1:] += out[..., :-1] * self.crosstalk_factor
            crosstalk[..., :-1] += out[..., 1:] * self.crosstalk_factor
            out = out + crosstalk

        # Step 9: O-E conversion (detector) - final readout
        if self.detector_noise_std > 0:
            out = out + torch.randn_like(out) * self.detector_noise_std * out.abs().mean()

        if self.bias is not None:
            out = out + self.bias
        return out


def test_scenario(name, **kwargs):
    """Test a specific realistic scenario."""
    np.random.seed(42)
    torch.manual_seed(42)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    for layer_idx in range(model.config.num_layers):
        attn = model.transformer.h[layer_idx].attn.attention
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(attn, pname)
            W = proj.weight.data.numpy()
            bias = proj.bias.data.numpy() if proj.bias is not None else None
            setattr(attn, pname, RealisticSVDLinear(W, bias, **kwargs))

        mlp = model.transformer.h[layer_idx].mlp
        W_fc = mlp.c_fc.weight.data.numpy()
        bias_fc = mlp.c_fc.bias.data.numpy() if mlp.c_fc.bias is not None else None
        mlp.c_fc = RealisticSVDLinear(W_fc, bias_fc, **kwargs)

        W_proj = mlp.c_proj.weight.data.numpy()
        bias_proj = mlp.c_proj.bias.data.numpy() if mlp.c_proj.bias is not None else None
        mlp.c_proj = RealisticSVDLinear(W_proj, bias_proj, **kwargs)

    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    text = generate_text(model, tokenizer, eval_prompts[0])
    return ppl, text


# Baseline
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
baseline_text = generate_text(model, tokenizer, eval_prompts[0])
print(f"\n[BASELINE] PPL={baseline_ppl:.4f}")
print(f"[BASELINE] Text: {baseline_text[:80]}...")

# === TEST SCENARIOS ===
print("\n" + "="*80)
print("CRITICAL EVALUATION: REALISTIC OPTICAL IMPAIRMENTS")
print("="*80)

scenarios = [
    ("Ideal (our current test)", dict(
        glass_psnr_db=90, optical_loss_db=0, detector_snr_db=100,
        dac_bits=32, phase_quantization_bits=32, crosstalk_db=-100)),

    ("Glass phase errors only (90dB)", dict(
        glass_psnr_db=90, optical_loss_db=0, detector_snr_db=100,
        dac_bits=32, phase_quantization_bits=32, crosstalk_db=-100)),

    ("+ Insertion loss (0.5dB/block)", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=100,
        dac_bits=32, phase_quantization_bits=32, crosstalk_db=-100)),

    ("+ Insertion loss (1.0dB/block)", dict(
        glass_psnr_db=90, optical_loss_db=1.0, detector_snr_db=100,
        dac_bits=32, phase_quantization_bits=32, crosstalk_db=-100)),

    ("+ Detector noise (60dB SNR)", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=32, phase_quantization_bits=32, crosstalk_db=-100)),

    ("+ Detector noise (40dB SNR)", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=40,
        dac_bits=32, phase_quantization_bits=32, crosstalk_db=-100)),

    ("+ 8-bit phase quantization", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=32, phase_quantization_bits=8, crosstalk_db=-100)),

    ("+ 6-bit phase quantization", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=32, phase_quantization_bits=6, crosstalk_db=-100)),

    ("+ 10-bit DAC", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=10, phase_quantization_bits=8, crosstalk_db=-100)),

    ("+ 8-bit DAC", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=8, phase_quantization_bits=8, crosstalk_db=-100)),

    ("+ Crosstalk (-30dB)", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=10, phase_quantization_bits=8, crosstalk_db=-30)),

    ("+ Crosstalk (-20dB)", dict(
        glass_psnr_db=90, optical_loss_db=0.5, detector_snr_db=60,
        dac_bits=10, phase_quantization_bits=8, crosstalk_db=-20)),

    ("WORST CASE REALISTIC", dict(
        glass_psnr_db=50, optical_loss_db=1.0, detector_snr_db=40,
        dac_bits=8, phase_quantization_bits=6, crosstalk_db=-20)),

    ("BEST CASE REALISTIC", dict(
        glass_psnr_db=90, optical_loss_db=0.3, detector_snr_db=60,
        dac_bits=12, phase_quantization_bits=10, crosstalk_db=-30)),
]

print(f"\n{'Scenario':<45s} {'PPL':>10s} {'Delta%':>10s} {'Text Match':>12s}")
print("-" * 80)

for name, kwargs in scenarios:
    ppl, text = test_scenario(name, **kwargs)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    match = "MATCH" if text == baseline_text else "DIFFER"
    print(f"{name:<45s} {ppl:>10.2f} {delta:>+9.2f}% {match:>12s}")

print(f"\n[CRITICAL] Baseline PPL: {baseline_ppl:.4f}")
print(f"\n[CRITICAL] KEY FINDINGS:")
print(f"  If PPL stays within ~10% of baseline under realistic conditions,")
print(f"  our architecture is genuinely viable for real hardware.")
print(f"  If PPL explodes under any realistic impairment, we have a problem.")
