import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

### Implementation
@dataclass
class GPTConfig:
    context_lengh: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # batched K, Q, V projections
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch
                                          .ones(config.context_lengh, config.context_lengh))
                                          .view(1, 1, config.context_lengh, config.context_lengh))

    def forward(self, inputs):
        # B: batch size, T: time dimension (sequence length), C: channel dimensions (embedding size)
        B, T, C = inputs.size()

        # calculate K, Q, V
        qkv = self.c_attn(inputs)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_heads, T, h_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_heads, T, h_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_heads, T, h_size)

        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v # (B, n_heads, T, T) x (B, n_heads, T, h_size) -> (B, n_heads, T, h_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        out = self.c_proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc1  = nn.Linear(n_embd, 4*n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*n_embd, n_embd)

    def forward(self, inputs):
        inputs = self.c_fc1(inputs)
        inputs = self.gelu(inputs)
        inputs = self.c_proj(inputs)
        return inputs

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FFN(config.n_embd)

    def forward(self, inputs):
        inputs = inputs + self.attn(self.ln_1(inputs)) # map operation
        inputs = inputs + self.mlp(self.ln_2(inputs))  # reduce operation
        return inputs

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_lengh, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)