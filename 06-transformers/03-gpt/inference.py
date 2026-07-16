import os
import torch
import tiktoken

from model import GPTConfig, GPTPretrained

logdir = "./logs/gpt-pretrained--20260708-024312/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tiktoken.get_encoding("gpt2")

text = "Once upon a time"
idx = torch.tensor(tokenizer.encode(text), device=device)[None, :]

ckpt_fname = os.path.join(logdir, "ckpt.pt")
checkpoint = torch.load(ckpt_fname, map_location=device)
gpt = GPTPretrained(GPTConfig(**checkpoint["model_config"]))
gpt.load_state_dict(checkpoint["model"])
gpt.to(device)

generated_tokens = gpt.generate(idx, seq_len=256, max_tokens=256).cpu().numpy().tolist()[0]
generated_text = tokenizer.decode(generated_tokens)
print(generated_text)
