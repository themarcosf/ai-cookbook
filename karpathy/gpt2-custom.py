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
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v # (B, n_heads, T, T) x (B, n_heads, T, h_size) -> (B, n_heads, T, h_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        out = self.c_proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, inputs):
        inputs = self.c_fc(inputs)
        inputs = self.gelu(inputs)
        inputs = self.c_proj(inputs)
        return inputs

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FFN(config)

    def forward(self, inputs):
        inputs = inputs + self.attn(self.ln_1(inputs)) # map operation
        inputs = inputs + self.mlp(self.ln_2(inputs))  # reduce operation
        return inputs

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_lengh, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, inputs, targets=None):
        B, T = inputs.size()
        assert T <= self.config.context_lengh, f'Cannot forward, model context length is exhausted. ' \
                                               f'Input has {T} tokens, but the maximum is {self.config.context_lengh}'
        
        # forward input embedding and positional embedding
        positions = torch.arange(0, T, dtype=torch.long, device=inputs.device)
        position_embeddings = self.transformer.wpe(positions)
        token_embeddings = self.transformer.wte(inputs)
        inputs = token_embeddings + position_embeddings
        
        # forward the transformer blocks
        for block in self.transformer.h:
            inputs = block(inputs)

        # forward the final layer norm and linear layer
        inputs = self.transformer.ln_f(inputs) 
        logits = self.lm_head(inputs)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT2 model weights from huggingface.co 🤗"""

        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        print(f'Loading weights from pretrained gpt: {model_type}')

        # create model config
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['context_lengh'] = 1024

        # create model with random weights
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ignore buffer keys of autoregressive masks

        # create model with pretrained weights
        from transformers import GPT2LMHeadModel
        pretrained_model = GPT2LMHeadModel.from_pretrained(model_type)
        pretrained_sd = pretrained_model.state_dict()

        # copy pretrained weights into randomly initialized model
        pretrained_sd_keys = pretrained_sd.keys()
        pretrained_sd_keys = [k for k in pretrained_sd_keys if not k.endswith('.attn.masked_bias')]
        pretrained_sd_keys = [k for k in pretrained_sd_keys if not k.endswith('.attn.bias')]

        #  OpenAI checkpoints use `Conv1D` modules from TensorFlow that needs to be transposed to match PyTorch
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(pretrained_sd_keys), f'Mismatched keys: {len(sd_keys)} vs {len(pretrained_sd_keys)}'

        for k in pretrained_sd_keys:
            if any(k.endswith(t) for t in transposed):
                # special treatment for `Conv1D` weights
                assert pretrained_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[k].T)
            else:
                # vanilla copy of weights
                assert pretrained_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[k])

        return model


# Example usage
if __name__ == '__main__':
    # Set the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    
    print(f'Using device: {device}')

    # Define the number of return sequences
    num_return_sequences = 5

    # Load the text data
    with open('../data/tiny_shakespear.txt', 'r') as f:
        text = f.read()

    # Load the tokenizer
    import tiktoken
    encodings = tiktoken.get_encoding('gpt2')

    # Encode the text data
    tokens = encodings.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Create input and target buffers
    context_length = 1024
    buffer = tokens.clone().detach()
    # num_batches = buffer.shape[0] // context_length
    num_batches = 4

    input_buffer = buffer[:num_batches * context_length]
    inputs = input_buffer.view(num_batches, context_length)

    target_buffer = buffer[1:num_batches * context_length + 1]
    targets = target_buffer.view(num_batches, context_length)

    x = inputs.to(device)
    y = targets.to(device)
    print('Tokens loaded')

    # Create new model
    model = GPT(GPTConfig())
    model.to(device)
    print('Model loaded')


    # Optimize the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Step {i+1}: Loss = {loss.item()}')
