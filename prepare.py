"""
Dataset preparation script: Tokenizes text into flat uint16 binary buffers (train.bin, val.bin)
using tiktoken (GPT-2 BPE) with fallback to character-level encoding.
"""
import os
import requests
import numpy as np

def prepare_dataset():
    data_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    input_file_path = os.path.join(data_dir, 'input.txt')
    
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading TinyShakespeare to {input_file_path}...")
        resp = requests.get(data_url)
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # Attempt tiktoken GPT-2 BPE encoding
    try:
        import tiktoken
        print("Tokenizing with tiktoken (gpt2)...")
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        vocab_size = enc.n_vocab
        print(f"tiktoken encoded: train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens (vocab size: {vocab_size})")
    except ImportError:
        print("tiktoken not installed; tokenizing at character level...")
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        train_ids = [stoi[c] for c in train_data]
        val_ids = [stoi[c] for c in val_data]
        print(f"char encoded: train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens (vocab size: {vocab_size})")

    # Export to uint16 flat binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_bin_path = os.path.join(data_dir, 'train.bin')
    val_bin_path = os.path.join(data_dir, 'val.bin')
    
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)
    print(f"Successfully saved binary buffers to {train_bin_path} and {val_bin_path}")

if __name__ == '__main__':
    prepare_dataset()
