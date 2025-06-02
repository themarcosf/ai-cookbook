"""
FineWeb-Edu dataset (for srs pretraining) with 10B tokens.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Downloads and tokenizes the data and saves data shards to disk.

Run as:
$ python gpt2-edu_fineweb10B.py

Will save shards to the local directory 'edu_fineweb10B_shards'.
"""
import multiprocessing as mp
import os

from datasets import load_dataset
import numpy as np
import tiktoken
from tqdm import tqdm

local_dir = 'edu_fineweb10B_shards'
remote_name = 'sample-10BT'
shard_size = int(1e8) # 100M tokens per shard

# create the local directory for data shards
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train')

# tokenization
encodings = tiktoken.get_encoding('gpt2')
eot_tkn = encodings._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot_tkn] 
    tokens.extend(encodings.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'Token dictionary tokens must be in range [0, 2^16)'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    with open(filename, 'wb') as f:
        f.write(tokens_np.tobytes())

nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)

            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index + 1}')

            progress_bar.update(len(tokens))
        else:
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f'fineweb_{split}_{shard_index:06d}.npy')

            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(DATA_CACHE_DIR, f'fineweb_{split}_{shard_index:06d}.npy')
        write_datafile(filename, all_tokens_np[:token_count])