import torch
import torch.nn as nn
from torch.nn import functional as F


### hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8

MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_ITERS = 200
NUM_EMBEDDINGS = 32


### data
with open('../data/tiny_shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()


### tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda string: [stoi[ch] for ch in string]
decode = lambda tokens: ''.join([itos[t] for t in tokens])

tokenized_text = torch.tensor(encode(text), dtype=torch.long)

n = int(len(tokenized_text) * 0.85)
train_dataset = tokenized_text[:n]
val_dataset = tokenized_text[n:]


### data loader
def get_batch(split, batch_size, verbose=False):
    data = train_dataset if split == 'train' else val_dataset
    if verbose:
        print('Data: ', data)
        print("Lenght of data: ", len(data))
    
    random_observations = torch.randint(0, len(data) - context_length, (batch_size,))
    if verbose:
        print("random_observations: ", random_observations)

    input_batch = torch.stack([data[obs:obs+context_length] for obs in random_observations])
    target_batch = torch.stack([data[obs+1:obs+context_length+1] for obs in random_observations])

    input_batch = input_batch.to(DEVICE)
    target_batch = target_batch.to(DEVICE)
    
    return input_batch, target_batch


### self attention head
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
        self.query = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
        self.value = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, inputs):
        B, T, C = inputs.shape

        k = self.key(inputs)
        q = self.query(inputs)

        # compute scaled attention scores, ie. affinities
        weights = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)

        # perform weighted aggregation of values
        v = self.value(inputs)
        logits = weights @ v

        return logits
    

### multi-head self attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(NUM_EMBEDDINGS, NUM_EMBEDDINGS)

    def forward(self, inputs):
        inputs = torch.cat([head(inputs) for head in self.heads], dim=-1)
        return self.projection(inputs)
    

### feed-forward layer
class FeedForward(nn.Module):
    inner_layer_dim = 4   # see https://arxiv.org/pdf/1706.03762, section 3.3

    def __init__(self, hidden_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, self.inner_layer_dim * hidden_size),
            nn.ReLU(),
            nn.Linear(self.inner_layer_dim * hidden_size, hidden_size)
        )

    def forward(self, inputs):
        return self.layer(inputs)
    

### residual connections
class Block(nn.Module):
    """ Transformer block: communication followed by computation. """

    def __init__(self, context_length, num_heads):
        super().__init__()
        head_size = context_length // num_heads
        self.self_attention_heads = MultiHeadSelfAttention(num_heads, head_size)
        self.feed_forward = FeedForward(context_length)
        self.layer_normalization_1 = nn.LayerNorm(context_length)
        self.layer_normalization_2 = nn.LayerNorm(context_length)

    def forward(self, inputs):
        inputs = inputs + self.self_attention_heads(self.layer_normalization_1(inputs))
        inputs = inputs + self.feed_forward(self.layer_normalization_2(inputs))
        return inputs


### Transformer model
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, NUM_EMBEDDINGS)
        self.position_embedding_table = nn.Embedding(context_length, NUM_EMBEDDINGS)
        self.blocks = nn.Sequential(*[Block(NUM_EMBEDDINGS, num_heads=4) for _ in range(4)])
        self.layer_normalization = nn.LayerNorm(NUM_EMBEDDINGS)
        self.language_model_head = nn.Linear(NUM_EMBEDDINGS, vocab_size)
        
    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        token_embeddings = self.token_embedding_table(inputs)                              
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE))
        contextual_embeddings = token_embeddings + position_embeddings                     
        contextual_embeddings = self.blocks(contextual_embeddings)
        logits = self.language_model_head(contextual_embeddings)                           

        if targets is None:
            loss = None
        else:
            _, _, C = logits.shape
            logits = logits.view(-1, C)  # Flatten to [B * T, C]
            targets = targets.view(-1)   # Flatten to [B * T]
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, num_predictions):
        """ `Inputs` is a tensor of shape [B, T], where T is the current context length. """
        for _ in range(num_predictions):
            current_context = inputs[:, -context_length:]
            logits, loss = self(current_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat([inputs, predicted_token], dim=1)

        return inputs


### training
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, y = get_batch(split, BATCH_SIZE)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

batch_size = 32
context_length = 8
model = TransformerModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
    if (iter + 1) % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'Iter {iter + 1} | Train loss {losses["train"]} | Val loss {losses["val"]}')

    X, y = get_batch('train', batch_size)
    logits, loss = model(X, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


### prediction
NUM_PREDICTIONS = 50
INPUTS = X[:1]

predicted_targets = model.generate(INPUTS, NUM_PREDICTIONS).squeeze(dim=0)
print('Predicted characters: ', decode(predicted_targets.tolist()))
