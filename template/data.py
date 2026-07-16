import torch
from dataclasses import dataclass


@dataclass
class DataConfig:
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2
    train_dir: str = ""
    eval_dir: str = ""

class DataModule:
    def __init__(self, config):
        self.config = config

        self.train_data = None
        self.eval_data = None
        self.train_dataloader = self.get_dataloader(True)
        self.eval_dataloader = self.get_dataloader(False)

    def get_dataloader(self, train: bool):
        return torch.utils.data.DataLoader(
                dataset=self.train_data if train else self.eval_data,
                batch_size=self.config.train_batch_size if train else self.config.eval_batch_size,
                shuffle=train,
                num_workers=self.config.num_workers)

    def get_batch(self, train: bool):
        data = self.train_data if train else self.eval_data
        bs = self.config.train_batch_size if train else self.config.eval_batch_size
        idx = torch.randint(0, len(data), (bs,))
        # ...


if __name__ == "__main__":
    data = DataModule(DataConfig())
    x, y = data.get_batch(train=True)
