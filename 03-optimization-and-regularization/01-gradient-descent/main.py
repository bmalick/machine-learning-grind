#!/home/malick/miniconda3/envs/pt/bin/python
import torch

import torch
import modules
import sgd

if __name__ == "__main__":
    data = modules.LinearDataset(w=torch.tensor([1.3, 0.6]),
                            b=torch.tensor([4.9]), n=1000, noise=0.01)
    dataloader = torch.utils.data.DataLoader(dataset=data, shuffle=True, batch_size=16, num_workers=4)
    print("num of batches: ", len(dataloader))
    model = modules.FromScratchLinearModel(num_inputs=2)
    optimizer = sgd.SGD(params=[model.w, model.b], lr=0.01)
    criterion = lambda x,y: ((x-y)**2 / 2).mean()

    print("Params to estimate are:")
    print("w:", dataloader.dataset.w, ", b:", dataloader.dataset.b)
    print("Before training:")
    print("w:", model.w.detach().numpy(), ", b:", model.b.item())
    sgd.train_model(model=model, data=dataloader, criterion=criterion, optimizer=optimizer, epochs=10)
    print("After training")
    print("w:", model.w.detach().numpy(), ", b:", model.b.item())