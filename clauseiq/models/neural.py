"""Models 2–4: neural token taggers trained from scratch (random init, no pretrained weights).

All three share the same input layer and output head and differ only in the encoder:

  input  : word embedding (vocab built on CUAD train)  ⊕ token-shape embedding
           ⊕ relative-document-position embedding ⊕ log-absolute-position embedding
  encoder: CNN          — stacked residual dilated 1-D convolutions (local n-gram patterns)
           BiLSTM       — 2-layer bidirectional LSTM (sequential context)
           BiLSTM+Attn  — BiLSTM followed by multi-head self-attention with a residual
                          connection (learns which context tokens matter for each token)
  output : per-token, per-category logits (41 independent sigmoids → multi-label tagging)

Spans are decoded from the token probabilities (see ``clauseiq.decoding``).
"""
from __future__ import annotations

import torch
from torch import nn

from clauseiq.text.tokenize import N_ABS_POS, N_REL_POS, N_SHAPES


class InputLayer(nn.Module):
    def __init__(self, vocab_size: int, word_dim: int = 128, feat_dim: int = 16, dropout: float = 0.25):
        super().__init__()
        self.word = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.shape = nn.Embedding(N_SHAPES, feat_dim, padding_idx=0)
        self.rel = nn.Embedding(N_REL_POS, feat_dim, padding_idx=0)
        self.abs = nn.Embedding(N_ABS_POS, feat_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.word_dropout = 0.05  # randomly replace tokens with <unk> during training
        self.out_dim = word_dim + 3 * feat_dim
        nn.init.normal_(self.word.weight, std=0.1)
        with torch.no_grad():
            self.word.weight[0].zero_()

    def forward(self, ids, shapes, rel, absp):
        if self.training and self.word_dropout > 0:
            drop = (torch.rand_like(ids, dtype=torch.float) < self.word_dropout) & (ids > 1)
            ids = ids.masked_fill(drop, 1)
        x = torch.cat([self.word(ids), self.shape(shapes), self.rel(rel), self.abs(absp)], dim=-1)
        return self.dropout(x)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(dim, 2 * dim, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):  # x: (B, T, D)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = nn.functional.glu(h, dim=-1)  # gated linear unit
        h = self.dropout(h) * mask.unsqueeze(-1)
        return self.norm(x + h)


class TokenTagger(nn.Module):
    """Shared wrapper; ``arch`` selects the encoder."""

    def __init__(self, arch: str, vocab_size: int, n_labels: int, word_dim: int = 128,
                 feat_dim: int = 16, hidden: int = 160, layers: int = 2, cnn_dim: int = 256,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1), heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        self.hparams = dict(arch=arch, vocab_size=vocab_size, n_labels=n_labels, word_dim=word_dim,
                            feat_dim=feat_dim, hidden=hidden, layers=layers, cnn_dim=cnn_dim,
                            dilations=list(dilations), heads=heads, dropout=dropout)
        self.inp = InputLayer(vocab_size, word_dim, feat_dim, dropout=0.25)
        d_in = self.inp.out_dim
        if arch == "cnn":
            self.proj = nn.Linear(d_in, cnn_dim)
            self.blocks = nn.ModuleList([ResidualDilatedBlock(cnn_dim, d, dropout=dropout * 0.7)
                                         for d in dilations])
            d_out = cnn_dim
        elif arch in ("bilstm", "bilstm_attn"):
            self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, bidirectional=True,
                                batch_first=True, dropout=dropout if layers > 1 else 0.0)
            d_out = 2 * hidden
            if arch == "bilstm_attn":
                self.attn = nn.MultiheadAttention(d_out, heads, dropout=0.1, batch_first=True)
                self.attn_norm = nn.LayerNorm(d_out)
                self.ffn = nn.Sequential(nn.Linear(d_out, 2 * d_out), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(2 * d_out, d_out))
                self.ffn_norm = nn.LayerNorm(d_out)
        else:
            raise ValueError(f"unknown arch {arch}")
        self.out_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_out, n_labels)
        nn.init.constant_(self.head.bias, -3.0)  # start from a low positive prior

    def forward(self, ids, shapes, rel, absp, return_attention: bool = False):
        mask = (ids != 0).float()
        x = self.inp(ids, shapes, rel, absp)
        attn_w = None
        if self.arch == "cnn":
            h = self.proj(x)
            for blk in self.blocks:
                h = blk(h, mask)
        else:
            h, _ = self.lstm(x)
            if self.arch == "bilstm_attn":
                a, attn_w = self.attn(h, h, h, key_padding_mask=(ids == 0),
                                      need_weights=return_attention, average_attn_weights=True)
                h = self.attn_norm(h + self.out_dropout(a))
                h = self.ffn_norm(h + self.ffn(h))
        logits = self.head(self.out_dropout(h))
        return (logits, attn_w) if return_attention else logits


def build_model(arch: str, **kw) -> TokenTagger:
    return TokenTagger(arch, **kw)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
