import torch
import torch.nn as nn
from torch.nn import functional as F

### hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8

MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
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


### bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        """
        `self.embedding` is 65 x 65, because for each of the 65 tokens in the vocabulary,
        we have a 65-dimensional vector that represents the probability of the token
        given the context.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, NUM_EMBEDDINGS)
        self.position_embedding_table = nn.Embedding(context_length, NUM_EMBEDDINGS)
        self.lm_head = nn.Linear(NUM_EMBEDDINGS, vocab_size)
        
    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        token_embeddings = self.token_embedding_table(inputs)                                # B, T, C
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE))  # T, C
        logits = self.lm_head(token_embeddings + position_embeddings)                        # B, T, V

        if targets is None:
            loss = None
        else:
            _, _, C = logits.shape
            logits = logits.view(-1, C)  # Flatten to [B * T, C]
            targets = targets.view(-1)   # Flatten to [B * T]
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, num_predictions):
        predictions = torch.zeros(inputs.shape[0] * num_predictions, dtype=torch.long)

        for i in range(num_predictions):
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            target = torch.multinomial(probs, num_samples=1)
            predictions[i] = target

        return predictions


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
model = BigramLanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'Iter {iter} | Train loss {losses["train"]} | Val loss {losses["val"]}')

    X, y = get_batch('train', batch_size)
    logits, loss = model(X, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


### prediction
NUM_PREDICTIONS = 50
INPUTS = X[:1]

predicted_targets = model.generate(INPUTS, NUM_PREDICTIONS)
print('Predicted characters: ', decode(predicted_targets.tolist()))
