# BDH-GPU core — adapted from ~/coding/baby_dragon_hatchling/src/bdh.py
#
# Changes from the original:
#   - forward() returns hidden states instead of language-model logits
#   - Removed lm_head and cross-entropy loss (handled by the classifier wrapper)
#   - Added encode() for efficient single-email feature extraction with chunking
#
# Architecture overview:
#   Token indices → Embedding → N BDHLayers → hidden states (B, T, D)
#
# Each BDHLayer runs:
#   1. Sparse encoding:  x ∈ R^D  →  Q = K = ReLU(x @ E)  ∈ R^{nh×N}
#   2. Causal attention: scores = QR @ QR^T / √N, capped to local window
#      Differential attention (Δ-Attn): output = (attn₁ − λ·attn₂) @ V
#   3. Synaptic state ρ accumulates across chunks (optional):
#        ρ ← forget_gate·ρ + QR^T @ V·scale
#      Cross-chunk contribution: output += QR @ ρ
#   4. Hebbian gating: y = x_sparse ⊙ y_sparse, decoded back to D
#
# Forget-gate modes for the recurrent synaptic state ρ:
#   "none"   — no forgetting (pure accumulation; fine for stateless training)
#   "scalar" — learned per-head bias
#   "data"   — input-dependent per-head gate (GLA-style; default for inference)

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 2
    n_embd: int = 128
    dropout: float = 0.1
    n_head: int = 4
    # Sparse dimension N = n_embd * mlp_internal_dim_multiplier // n_head
    # N=1024 at default settings (128*32//4). Sweet spot for quality/speed.
    mlp_internal_dim_multiplier: int = 32
    vocab_size: int = 256          # byte-level tokenisation
    max_seq_len: int = 1024        # longest email chunk we process at once
    forget_mode: str = "none"      # "none" | "scalar" | "data"
    diff_attn: bool = True         # Differential Attention (Δ-Attn)
    attn_window: int = 64          # local attention window (0 = full causal)


# ─── RoPE helpers ────────────────────────────────────────────────────────────

def _build_rope_freqs(N: int, max_len: int) -> torch.Tensor:
    """Precompute complex RoPE frequencies: shape (1, 1, max_len, N//2)."""
    raw = 1.0 / (2 ** 16 ** (torch.arange(0, N, 1, dtype=torch.float32) / N)) / (2 * math.pi)
    q2 = (raw / 2).floor() * 2  # quantize(q=2) — pairs share frequency
    angles = (
        torch.arange(0, max_len, dtype=torch.float32).view(1, 1, -1, 1) * q2.view(1, 1, 1, N)
    ) % 1 * (2 * math.pi)
    cos_h = torch.cos(angles[:, :, :, 0::2])
    sin_h = torch.sin(angles[:, :, :, 0::2])
    return torch.complex(cos_h, sin_h)  # (1, 1, max_len, N//2)


# ─── Attention ───────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """BDH attention with optional recurrent synaptic state.

    Stateless mode (state=None): standard parallel causal attention.
    Stateful mode  (state≠None): adds cross-chunk contribution from ρ.

    State shape: (B, n_head, N, D) — accumulated outer products of QR and V.
    """

    def __init__(self, config: BDHConfig) -> None:
        super().__init__()
        self.config = config
        nh = config.n_head
        D  = config.n_embd
        N  = config.mlp_internal_dim_multiplier * D // nh

        self._scale = N ** -0.5
        self._forget_mode = config.forget_mode
        self._diff_attn   = config.diff_attn
        self._attn_window = config.attn_window

        # RoPE — precomputed complex exponentials, shape (1, 1, max_len, N//2)
        self._freq_cplx = nn.Buffer(_build_rope_freqs(N, config.max_seq_len))

        # Forget gate parameters
        if config.forget_mode == "scalar":
            self.forget_bias = nn.Parameter(torch.zeros(1, nh, 1, 1))
        elif config.forget_mode == "data":
            self.forget_proj = nn.Linear(D, nh, bias=True)
            nn.init.zeros_(self.forget_proj.weight)
            nn.init.zeros_(self.forget_proj.bias)

        # Differential attention: one λ per head-pair, init 0 → sigmoid(0) = 0.5
        if config.diff_attn:
            assert nh % 2 == 0, f"diff_attn requires even n_head, got {nh}"
            self.lambda_param = nn.Parameter(torch.zeros(1, nh // 2, 1, 1))

    def _rope(self, v: torch.Tensor, T: int, offset: int) -> torch.Tensor:
        freq = self._freq_cplx[:, :, offset : offset + T, :]
        vc = torch.view_as_complex(v.float().reshape(*v.shape[:-1], -1, 2))
        return torch.view_as_real(vc * freq).reshape(*v.shape).to(v.dtype)

    def forward(
        self,
        Q: torch.Tensor,
        V: torch.Tensor,
        state: Optional[torch.Tensor],
        pos_offset: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            Q:   (B, n_head, T, N)  sparse ReLU activations (K = Q)
            V:   (B, 1, T, D)       raw embeddings broadcast across heads
            state: (B, n_head, N, D) or None
            pos_offset: absolute position of the first token (for RoPE)

        Returns:
            output:    (B, n_head, T, D)
            new_state: (B, n_head, N, D) or None
        """
        _, _, T, N = Q.size()
        QR = self._rope(Q, T, offset=pos_offset)

        # Build causal (+ optional local window) mask
        if 0 < self._attn_window < T:
            mask = torch.ones(T, T, device=Q.device, dtype=Q.dtype).tril(diagonal=-1)
            mask = mask.triu(diagonal=-(self._attn_window - 1))
        else:
            mask = None

        # ── Within-chunk attention ──
        if self._diff_attn:
            B   = Q.size(0)
            ng  = Q.size(1) // 2           # number of diff groups
            QR1 = QR[:, 0::2]              # even heads  (B, ng, T, N)
            QR2 = QR[:, 1::2]              # odd heads
            if mask is not None:
                s1 = (QR1 @ QR1.mT * self._scale) * mask
                s2 = (QR2 @ QR2.mT * self._scale) * mask
            else:
                s1 = (QR1 @ QR1.mT * self._scale).tril(diagonal=-1)
                s2 = (QR2 @ QR2.mT * self._scale).tril(diagonal=-1)
            lam    = torch.sigmoid(self.lambda_param)
            scores = s1 - lam * s2                                  # (B, ng, T, T)
            Vg     = V.expand(B, ng, T, -1) if V.size(1) == 1 else V[:, :ng]
            out    = (scores @ Vg).repeat_interleave(2, dim=1)      # (B, nh, T, D)
        else:
            if mask is not None:
                scores = (QR @ QR.mT * self._scale) * mask
            else:
                scores = (QR @ QR.mT * self._scale).tril(diagonal=-1)
            out = scores @ V                                        # (B, nh, T, D)

        # ── Forget gate ──
        if self._forget_mode == "scalar":
            fg = torch.sigmoid(self.forget_bias)
        elif self._forget_mode == "data":
            x_mean = V.squeeze(1).mean(dim=1)                      # (B, D)
            fg = torch.sigmoid(self.forget_proj(x_mean)).unsqueeze(-1).unsqueeze(-1)
        else:
            fg = None

        # ── Cross-chunk contribution ──
        if state is not None:
            gated = fg * state if fg is not None else state
            out   = out + QR @ gated
            new_state = gated + QR.transpose(-2, -1) @ V * self._scale
        else:
            new_state = None

        return out, new_state


# ─── BDHLayer ────────────────────────────────────────────────────────────────

class BDHLayer(nn.Module):
    """Single BDH computation layer with per-layer encoder/decoder parameters."""

    def __init__(self, config: BDHConfig, attn: Attention) -> None:
        super().__init__()
        nh = config.n_head
        D  = config.n_embd
        N  = config.mlp_internal_dim_multiplier * D // nh

        self.attn      = attn
        self.ln        = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop      = nn.Dropout(config.dropout)
        self.config    = config

        # Per-layer learnable sparse projections
        self.encoder   = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.decoder   = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))

    def forward(
        self,
        x: torch.Tensor,                   # (B, 1, T, D)
        state: Optional[torch.Tensor],
        pos_offset: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        C = self.config
        B, _, T, D = x.size()
        nh = C.n_head
        N  = D * C.mlp_internal_dim_multiplier // nh

        # Sparse encoding: flat matmul is 2× faster than broadcast batched matmul
        enc_flat = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
        x_latent = (x.squeeze(1).reshape(B * T, D) @ enc_flat).view(B, T, nh, N).permute(0, 2, 1, 3)
        x_sparse = F.relu(x_latent)                                # (B, nh, T, N)

        out, new_state = self.attn(Q=x_sparse, V=x, state=state, pos_offset=pos_offset)
        out = self.ln(out)

        # Encode_v: einsum is 40% faster than batched matmul
        y_latent = torch.einsum("bhtd,hdn->bhtn", out, self.encoder_v)
        y_sparse = F.relu(y_latent)

        # Hebbian gating (the core of the architecture — do NOT change to additive/max)
        xy_sparse = self.drop(x_sparse * y_sparse)

        # Decode back to D-space
        y_mlp = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        return self.ln(x + self.ln(y_mlp)), new_state


# ─── BDH (feature extractor) ─────────────────────────────────────────────────

class BDH(nn.Module):
    """BDH language encoder — returns hidden states (B, T, D).

    Unlike the original, this class does NOT include lm_head or loss computation.
    It is a pure feature extractor; the classification head lives in BDHSpamClassifier.

    Supports two training/inference modes:
      Stateless (state=None):  each forward pass is independent. Fast, simple.
      Stateful  (state≠None):  state carries over between chunks for continuous
                               Hebbian learning across long emails or sessions.
    """

    def __init__(self, config: BDHConfig, use_grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.config = config
        self.use_grad_checkpoint = use_grad_checkpoint

        # Shared attention module (topology shared, encoder/decoder are per-layer)
        self.attn  = Attention(config)
        self.ln    = nn.LayerNorm(config.n_embd, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop  = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            BDHLayer(config, self.attn) for _ in range(config.n_layer)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                          # (B, T)
        state: Optional[list[torch.Tensor]] = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """
        Returns:
            hidden: (B, T, D)   — per-token representations
            new_state: per-layer state list (same length as self.layers)
        """
        x = self.ln(self.embed(idx)).unsqueeze(1)   # (B, 1, T, D)

        new_state: list[Optional[torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None
            if self.use_grad_checkpoint and self.training:
                x, s = checkpoint(layer, x, layer_state, pos_offset, use_reentrant=False)
            else:
                x, s = layer(x, layer_state, pos_offset)
            new_state.append(s)

        B, _, T, D = x.size()
        return x.view(B, T, D), new_state

    def init_state(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Initialise zero synaptic state for stateful inference."""
        cfg = self.config
        N   = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
        return [
            torch.zeros(batch_size, cfg.n_head, N, cfg.n_embd,
                        device=device, dtype=torch.float32)
            for _ in range(cfg.n_layer)
        ]
