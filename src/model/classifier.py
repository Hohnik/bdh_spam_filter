# BDHSpamClassifier — BDH feature encoder + binary classification head.
#
# Two distinct roles of the BDH recurrent synaptic state ρ:
#
#   Role A — Within-email chunking (computational necessity)
#     BDH attention is O(T²). Emails longer than chunk_size must be split.
#     ρ carries context across chunks so no information is discarded.
#     A fresh zero-state is used for EVERY classification call.
#
#   Role B — Cross-session continuous learning (the architecture's key property)
#     After initial training the model holds a *persistent* inference state ρ_live
#     that accumulates Hebbian updates as it processes new confirmed spam/ham.
#     This state is owned by SpamFilter (not by this module) and is passed in
#     explicitly via the `live_state` argument to predict().
#     This is what prevents the model from drifting stale as spammers evolve.
#
# The distinction matters: Role A uses a transient per-call state; Role B uses
# a persistent cross-call state managed by the inference layer (SpamFilter).

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .bdh import BDH, BDHConfig


@dataclasses.dataclass
class SpamClassifierConfig:
    # BDH core — sized to be fast on a home-server CPU
    n_layer: int = 2
    n_embd: int = 128
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 32   # N = 128*32//4 = 1024 per head
    dropout: float = 0.1
    vocab_size: int = 256                   # byte-level, no external tokeniser
    forget_mode: str = "none"               # "none" for offline training
    diff_attn: bool = True
    attn_window: int = 64

    # Chunking / position encoding — two SEPARATE concerns:
    #
    #   chunk_size   — tokens per forward pass (O(T²) attention, so keep this
    #                  small enough to fit in RAM).  Typical: 512.
    #
    #   max_position — size of the RoPE frequency table; must be ≥ the maximum
    #                  absolute position we'll ever encode, i.e. ≥ max_email_bytes.
    #                  Governs how many unique positions the model can represent.
    #
    # Keeping these separate avoids the bug where pos_offset = chunk_size falls
    # outside the RoPE table when processing the second chunk.
    chunk_size:    int = 512
    max_position:  int = 4096   # = max_email_bytes; RoPE table size
    max_email_bytes: int = 4096

    use_grad_checkpoint: bool = False

    def to_bdh_config(self) -> BDHConfig:
        return BDHConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            mlp_internal_dim_multiplier=self.mlp_internal_dim_multiplier,
            dropout=self.dropout,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_position,  # RoPE table covers ALL possible positions
            forget_mode=self.forget_mode,
            diff_attn=self.diff_attn,
            attn_window=self.attn_window,
        )


class BDHSpamClassifier(nn.Module):
    """BDH-based binary spam classifier.

    Architecture:
        byte tokens → BDH encoder → [chunk-aware mean pool] → Linear(D→1) → sigmoid

    For emails that fit within chunk_size (the common case) the forward pass is a
    single call with no state overhead. Longer emails are transparently chunked.
    """

    def __init__(self, config: SpamClassifierConfig) -> None:
        super().__init__()
        self.config  = config
        self.encoder = BDH(
            config.to_bdh_config(),
            use_grad_checkpoint=config.use_grad_checkpoint,
        )
        self.cls_head = nn.Linear(config.n_embd, 1)
        nn.init.normal_(self.cls_head.weight, std=0.02)
        nn.init.zeros_(self.cls_head.bias)

    # ── Core encoding ─────────────────────────────────────────────────────────

    def _encode(
        self,
        token_ids: torch.Tensor,                        # (B, T)
        initial_state: Optional[list[torch.Tensor]],   # Role B persistent state, or None
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """Encode token_ids into a (B, D) document embedding.

        If T ≤ chunk_size: single forward pass, no state overhead.
        If T  > chunk_size: chunked forward pass, ρ carries context between chunks.

        The *initial_state* argument is for Role B (cross-session learning).
        When None, a fresh zero state is created (Role A / training).

        Returns:
            pooled:    (B, D) — mean-pooled hidden states across the whole email
            end_state: per-layer state after the last chunk (for Role B callers)
        """
        B, T     = token_ids.size()
        device   = token_ids.device
        chunk_sz = self.config.chunk_size

        if initial_state is None:
            state: list[Optional[torch.Tensor]] = self.encoder.init_state(B, device)
        else:
            state = initial_state  # type: ignore[assignment]

        all_hidden: list[torch.Tensor] = []
        pos = 0
        while pos < T:
            chunk          = token_ids[:, pos : pos + chunk_sz]
            hidden, state  = self.encoder(chunk, state=state, pos_offset=pos)
            all_hidden.append(hidden)                   # (B, chunk_len, D)
            pos += chunk_sz

        pooled = torch.cat(all_hidden, dim=1).mean(dim=1)  # (B, D)
        return pooled, state

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids: torch.Tensor,                        # (B, T)
        labels: Optional[torch.Tensor] = None,          # (B,) float 0/1
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Training forward: uses fresh zero state (Role A only).

        Returns:
            logits: (B, 1)
            loss:   BCE scalar, or None
        """
        pooled, _ = self._encode(token_ids, initial_state=None)
        logits    = self.cls_head(pooled)
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(1), labels.float()
            )
        return logits, loss

    @torch.no_grad()
    def predict(
        self,
        token_ids: torch.Tensor,                        # (B, T)
        live_state: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """Inference forward.

        Args:
            token_ids:  (B, T) byte values 0-255
            live_state: Role B persistent state from previous calls (or None for
                        a fresh zero state, which is equivalent to stateless inference)

        Returns:
            probs:     (B,) spam probabilities in [0, 1]
            new_state: updated state to hand back to the next predict() call
        """
        pooled, new_state = self._encode(token_ids, initial_state=live_state)
        probs = torch.sigmoid(self.cls_head(pooled).squeeze(1))
        return probs, new_state

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls, path: str, device: Optional[torch.device] = None
    ) -> "BDHSpamClassifier":
        if device is None:
            device = torch.device("cpu")
        ckpt   = torch.load(path, map_location=device, weights_only=True)
        config = SpamClassifierConfig(**ckpt["config"])
        model  = cls(config)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return model

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "config":      dataclasses.asdict(self.config),
                "model_state": self.state_dict(),
            },
            path,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
