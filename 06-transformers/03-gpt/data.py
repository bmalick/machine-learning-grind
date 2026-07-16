import torch
import tiktoken
import numpy as np
from dataclasses import dataclass

@dataclass
class DataConfig:
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2
    seq_len: int = 256

class TinyStoriesData:
    def __init__(self, config):
        self.config = config
        self.train_data = np.memmap("./train-data.dat", dtype=np.uint16, mode='r')
        self.eval_data = np.memmap("./eval-data.dat", dtype=np.uint16, mode='r')
        self.config = config

    def get_batch(self, split):
        data = self.train_data if split=="train" else self.eval_data
        bs = self.config.train_batch_size if split=="train" else self.config.eval_batch_size
        seq_len = self.config.seq_len
        idx = torch.randint(0, len(data)-seq_len, (bs,))
        x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in idx])
        return x, y


if __name__ == "__main__":
    data = TinyStoriesData(DataConfig())
    tokenizer = tiktoken.get_encoding("gpt2")
    x, y = data.get_batch("train")
    for i in range(len(x)):
        print("input:", tokenizer.decode(x[i].tolist()))
        print("target:", tokenizer.decode(y[i].tolist()))
        print("="*90)
