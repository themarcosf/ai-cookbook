import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import transformer_lens

os.chdir('..')

with open(os.path.expanduser('~/.huggingface/token')) as f:
    os.environ['HF_TOKEN'] = f.read().strip()

checkpoint = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'

hf_model = DistilBertForSequenceClassification.from_pretrained(
    Path(checkpoint).resolve(),
    torch_dtype=torch.float32,
    token=os.environ.get("HF_TOKEN", "") 
)

model = transformer_lens.HookedTransformer.from_pretrained(
    checkpoint, 
    hf_model=hf_model,
    fold_ln=False, 
    center_writing_weights=False, 
    center_unembed=False, 
    fold_value_biases=False
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')

inputs = tokenizer(
  'I love this movie!', 
  return_tensors='pt', 
  padding='max_length', 
  truncation=True,
  max_length=512
)

with torch.no_grad():
    logits = model('I love this movie!', return_type='logits')
    probs = F.softmax(logits, dim=-1)
    predicted_class_id = probs[:, -1, :].argmax().item()
    confidence_score = probs[:, -1, predicted_class_id].item()
    breakpoint()
    predicted_label = hf_model.config.id2label[predicted_class_id]

print({'label': predicted_label, 'score': confidence_score})