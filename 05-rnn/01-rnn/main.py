import os
import re
import math
import json
import torch
import torchvision
import collections
from torch import nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)

# ----------------- Data -----------------

@dataclass
class DataConfig:
    train_batch_size: int = 1024
    eval_batch_size: int = 1024
    num_workers: int = 2
    num_steps: int = 32
    num_train: int = 10000
    num_eval: int = 5000

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = list(sorted(set(["<unk>"] + reserved_tokens + [t for t,f in self.token_freqs if f>min_freq])))
        self.token_to_idx = {t:i for i,t in enumerate(self.idx_to_token)}

    def __len__(self): return len(self.token_to_idx)

    @property
    def unk(self): return self.token_to_idx["<unk>"]

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(t) for t in tokens]
    
    def to_tokens(self, indices):
        if hasattr(indices, "__len__") and len(indices)>1:
            return [self.idx_to_token[int(i)] for i in indices]
        return self.idx_to_token[indices]

class DataModule:
    def __init__(self, config):
        self.config = config
        with open("timemachine.txt", 'r') as f:
            text = f.read()
        text = re.sub("[^A-Za-z]+", ' ', text).lower()

        words = text.split()
        uni_vocab = Vocab(tokens=words)
        freqs = [f for _,f in uni_vocab.token_freqs]
        plt.loglog(freqs)
        plt.title("freqs")
        plt.savefig("zipf-law.jpg")
        plt.close()

        bigrams = ["--".join(p) for p in zip(words[:-1], words[1:])]
        trigrams = ["--".join(t) for t in zip(words[:-2], words[1:-1], words[2:])]
        bigram_vocab = Vocab(bigrams)
        trigram_vocab = Vocab(trigrams)
        plt.loglog([f for _,f in uni_vocab.token_freqs], label="unigram")
        plt.loglog([f for _,f in bigram_vocab.token_freqs], label="bigram")
        plt.loglog([f for _,f in trigram_vocab.token_freqs], label="trigram")
        plt.legend()
        plt.savefig("zipf-law-ngrams.jpg")
        plt.close()

        tokens = list(text)
        vocab = Vocab(tokens=tokens)
        corpus = [vocab[token] for token in tokens]
        array = torch.tensor([corpus[i:i+config.num_steps+1] for i in range(len(corpus)-config.num_steps)])
        self.X = array[:,:-1]
        self.y = array[:,1:]

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.train_dataloader = self.get_dataloader(True)
        self.eval_dataloader = self.get_dataloader(False)

    def get_dataloader(self, train: bool):
        idx = slice(0, self.config.num_train) if train else slice(self.config.num_train, self.config.num_train+self.config.num_eval)
        return torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(self.X[idx], self.y[idx]),
                batch_size=self.config.train_batch_size if train else self.config.eval_batch_size,
                shuffle=train,
                num_workers=self.config.num_workers)


# ----------------- Model -----------------

@dataclass
class ModuleConfig:
    num_blocks: int = 2
    num_hiddens: int = 32
    vocab_size: int = 32
    sigma: float = 0.01
    num_inputs: int = 4580

class RNNfromScratch(nn.Module): # RNN cell
    def __init__(self, config):
        super(RNNfromScratch, self).__init__()
        self.num_hiddens = config.num_hiddens
        self.sigma = config.sigma
        self.W_xh = nn.Parameter(torch.randn((config.num_inputs, config.num_hiddens)) * config.sigma)
        self.W_hh = nn.Parameter(torch.randn(config.num_hiddens, config.num_hiddens) * config.sigma)
        self.b_h = nn.Parameter(torch.zeros(config.num_hiddens))
    
    def forward(self, inputs, state=None):
        # inputs shape is: (num_steps, batch_size, num_inputs)
        if state is None:
            # inputs.size(1) is batch_size
            state = torch.zeros((inputs.size(1), self.num_hiddens), device=inputs.device)
        else: state, = state
        outputs = []
        for x in inputs: # shape of inputs: (num_steps, batch_size, num_inputs)
            # for each step
            # print(f"sequence {len(outputs)} shape:", x.shape)
            state = torch.tanh(torch.matmul(x, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

class RNNLMfromScratch(nn.Module):
    def __init__(self, config):
        super(RNNLMfromScratch, self).__init__()
        self.config = config
        rnn = RNNfromScratch(config)
        self.W_hq = nn.Parameter(torch.randn(rnn.num_hiddens, config.vocab_size) * rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(config.vocab_size))
        self.vocab_size = config.vocab_size
        self.rnn = rnn

    def compute_loss(self, logits, targets):
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    
    def one_hot(self, x):
        # x shape: (batch_size, num_steps)
        # output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(x.T, self.vocab_size).type(torch.float32)

    def forward(self, x, state=None, targets=None):
        embeddings = self.one_hot(x)
        # print(embeddings.shape)
        rnn_outputs, state = self.rnn(embeddings, state)
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        # for o in outputs: print(o.shape); break
        out = torch.stack(outputs, dim=1)
        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: (B,T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx, None) # (B, T, C)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# ----------------- Train -----------------
def grad_clip(clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > clip_val:
        for p in params:
            p.grad[:] *= clip_val / norm

@dataclass
class TrainConfig:
    run_name: str = "run"
    max_epochs: int = 100
    eval_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1.
    clip_val: float = 1.
    save_model: bool = True
    figsize: tuple[float, float] = (8., 4.5)
    figlog: bool = False
    figgrid: bool = True

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{self.run_name}--{timestamp}"
        self.logdir = os.path.join("logs", self.run_id)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        self.model_save_fname = os.path.join(self.logdir, self.run_name+".pth")

class Trainer:
    def __init__(self, config, datamodule, model):
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.writer = SummaryWriter(log_dir=config.logdir)

    def to_device(self, batch):
        return [a.to(self.config.device) for a in batch]

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.config.learning_rate)

    def configure_metrics(self):
        self.metric_names = ["acc"]
        self.metric_funcs = [
            lambda x,y: (x.argmax(dim=-1)==y).float().mean()
        ]
        self.perstep_metrics = {n: {"train": [], "eval": []} for n in self.metric_names}
        self.perepoch_metrics = {n: {"train": [], "eval": []} for n in self.metric_names}

    def compute_metrics(self, y_hat, y_true):
        assert len(self.metric_names)==len(self.metric_funcs)
        out = {}
        if len(self.metric_names)==0: return {}
        for n, func in zip(self.metric_names, self.metric_funcs):
            out[n] = func(y_hat, y_true).item()
        return out

    def save_model(self):
        if self.config.save_model:
            torch.save(self.model.state_dict(), self.config.model_save_fname)
            print(f"Model saved ad {self.config.model_save_fname}")

    def fit(self):
        self.model = self.model.to(self.config.device)
        self.configure_optimizers()
        self.configure_metrics()

        self.current_epoch = 0
        self.train_steps = 0
        self.eval_steps = 0

        self.perstep_losses = {"train": [], "eval": []}
        self.perepoch_losses = {"train": [], "eval": []}

        for _ in range(self.config.max_epochs):
            self.train_step()
            self.eval_step()

            if self.current_epoch % self.config.eval_interval == 0:
                metrics_str = " | ".join([f"train_{k}: {v['train'][-1]:.5f} | eval_{k}: {v['eval'][-1]:.5f}" for k, v in self.perepoch_metrics.items()])
                print(
                    f"Epoch: {self.current_epoch:3d} | "
                    f"train_loss: {self.perepoch_losses['train'][-1]:.5f} | "
                    f"eval_loss: {self.perepoch_losses['eval'][-1]:.5f} | "
                    f"{metrics_str}"
                )

            self.current_epoch += 1

        self.writer.close()
        self.save_model()
        self.make_plots()
        self.save_logs()

    def train_step(self):
        self.model.train()
        
        epoch_loss = 0.
        num_instances = 0
        epoch_metrics = {n: 0. for n in self.metric_names}

        for batch in self.datamodule.train_dataloader:
            batch = self.to_device(batch)
            out, loss = self.model(*batch[:-1], state=None, targets=batch[-1])

            self.optimizer.zero_grad()
            loss.backward()
            grad_clip(self.config.clip_val, self.model)

            self.optimizer.step()

            bs = batch[-1].size(0)
            num_instances += bs
            epoch_loss += loss.item() * bs

            metrics = self.compute_metrics(out, batch[-1])
            for k,v in metrics.items():
                self.perstep_metrics[k]["train"].append(v)
                self.writer.add_scalar(f"perstep_{k}/train", v, self.train_steps)
                epoch_metrics[k] += v * bs

            self.perstep_losses["train"].append(loss.item())
            self.writer.add_scalar("perstep_loss/train", loss.item(), self.train_steps)


            self.train_steps += 1
        
        for k,v in epoch_metrics.items():
            self.perepoch_metrics[k]["train"].append(v/num_instances )
            self.writer.add_scalar(f"perepoch_{k}/train", v/num_instances, self.current_epoch)

        epoch_loss /= num_instances
        self.perepoch_losses["train"].append(epoch_loss)
        self.writer.add_scalar("perepoch_loss/train", epoch_loss, self.current_epoch)


    def eval_step(self):
        self.model.eval()

        epoch_loss = 0.
        num_instances = 0
        epoch_metrics = {n: 0. for n in self.metric_names}

        for batch in self.datamodule.eval_dataloader:
            batch = self.to_device(batch)
            with torch.no_grad():
                out, loss = self.model(*batch[:-1], state=None, targets=batch[-1])

            bs = batch[-1].size(0)
            num_instances += bs
            epoch_loss += loss.item() * bs

            metrics = self.compute_metrics(out, batch[-1])
            for k,v in metrics.items():
                self.perstep_metrics[k]["eval"].append(v)
                self.writer.add_scalar(f"perstep_{k}/eval", v, self.eval_steps)
                epoch_metrics[k] += v * bs

            self.perstep_losses["eval"].append(loss.item())
            self.writer.add_scalar("perstep_loss/eval", loss.item(), self.eval_steps)

            self.eval_steps += 1
        
        for k,v in epoch_metrics.items():
            self.perepoch_metrics[k]["eval"].append(v/num_instances )
            self.writer.add_scalar(f"perepoch_{k}/eval", v/num_instances, self.current_epoch)

        epoch_loss /= num_instances
        self.perepoch_losses["eval"].append(epoch_loss)
        self.writer.add_scalar("perepoch_loss/eval", epoch_loss, self.current_epoch)

    def make_plots(self):
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for split, values in self.perepoch_losses.items():
            if self.config.figlog:
                ax.semilogy(values, label=split, linestyle="-" if split=="train" else "--")
            else:
                ax.plot(values, label=split, linestyle="-" if split=="train" else "--")
            if self.config.figgrid: ax.grid()
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_title("loss")
        fig.savefig(os.path.join(self.config.logdir, "loss.jpg"))
        plt.close()

        for name,values in self.perepoch_metrics.items():
            fig, ax = plt.subplots(figsize=self.config.figsize)
            for split, v in values.items():
                if self.config.figlog:
                    ax.semilogy(v, label=split, linestyle="-" if split=="train" else "--")
                else:
                    ax.plot(v, label=split, linestyle="-" if split=="train" else "--")
                if self.config.figgrid: ax.grid()
            ax.legend()
            ax.set_xlabel("epochs")
            ax.set_title(name)
            fig.savefig(os.path.join(self.config.logdir, f"{name}.jpg"))
            plt.close()

    def save_logs(self):
        fname = os.path.join(self.config.logdir, "losses.json")
        with open(fname, "w") as f:
            json.dump({"perstep": self.perstep_losses, "perepoch": self.perepoch_losses}, f)
        print(f"Save losses at {fname}")

        fname = os.path.join(self.config.logdir, "metrics.json")
        with open(fname, "w") as f:
            json.dump({"perstep": self.perstep_metrics, "perepoch": self.perepoch_metrics}, f)
        print(f"Save metrics at {fname}")

if __name__ == "__main__":
    # !wget -O timemachine.txt http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt

    datamodule = DataModule(DataConfig())
    module = RNNLMfromScratch(ModuleConfig(num_inputs=datamodule.vocab_size, vocab_size=datamodule.vocab_size))
    trainer = Trainer(TrainConfig(), datamodule, module)
    trainer.fit()
    idx = torch.tensor([datamodule.vocab[list("it has")]]).to(trainer.config.device)
    out = module.generate(idx, max_new_tokens=20)[0]
    print("Test inference:", "".join(datamodule.vocab.idx_to_token[i] for i in out))
