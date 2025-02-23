import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 32
CONTEXT_LENGTH = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_ITERS = 200

# data
with open('../data/tiny_shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda string: [stoi[ch] for ch in string]
decode = lambda tokens: ''.join([itos[t] for t in tokens])

tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# split data into train and validation
n = int(len(tokenized_text) * 0.85)
train_dataset = tokenized_text[:n]
val_dataset = tokenized_text[n:]

# data loader
def get_batch(split):
    data = train_dataset if split == 'train' else val_dataset
    
    random_observations = torch.randint(0, len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))

    input_batch = torch.stack([data[obs:obs+CONTEXT_LENGTH] for obs in random_observations])
    target_batch = torch.stack([data[obs+1:obs+CONTEXT_LENGTH+1] for obs in random_observations])

    input_batch = input_batch.to(DEVICE)
    target_batch = target_batch.to(DEVICE)
    
    return input_batch, target_batch

# loss estimation
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, inputs, targets=None):
        logits = self.embedding(inputs)

        if targets is None:
            loss = None
        else:
            _, _, C = logits.shape
            logits = logits.view(-1, C) 
            targets = targets.view(-1)  
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
    
# training
model = BigramLanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for i in range(MAX_ITERS):
    if (i + 1) % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(f'iter: {i+1}, train_loss: {losses["train"]:.4f}, val_loss: {losses["val"]:.4f}')
    
    x, y = get_batch('train')
    _, loss = model(x, y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# prediction
NUM_PREDICTIONS = 50
INPUTS = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
predictions = model.generate(INPUTS, NUM_PREDICTIONS)
print(decode(predictions.tolist()))