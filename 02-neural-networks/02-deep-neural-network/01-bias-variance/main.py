#!//home/malick/miniconda3/envs/pt/bin/python3
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def target_func(x):
    return np.exp(np.sin(2*np.pi*x))

def make_sinusoidal_dataset(n: int=100, noise: float=0.1, plot=False):
    X = np.random.uniform(low=0, high=1, size=(n,1))
    noise = np.random.normal(loc=0., scale=noise, size=(n,1))
    y = target_func(X) + noise
    if plot:
        plt.scatter(X,y, s=5); plt.show()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    data_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X,y),
        batch_size=32, shuffle=True
    )
    return data_loader

def shallow_model_for_sin(num_hidden_units: int):
    return nn.Sequential(
        nn.Linear(in_features=1, out_features=num_hidden_units),
        nn.Tanh(),
        nn.Linear(in_features=num_hidden_units, out_features=1),
    )

def train(ax, n, num_hidden_units=3, lr=0.01, num_epochs=100):
    print(f"##### n: {n}, num_hidden_units: {num_hidden_units}")
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_loader = make_sinusoidal_dataset(n=n, noise=0.1)
    model = shallow_model_for_sin(num_hidden_units=num_hidden_units).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    for e in range(num_epochs):
        epoch_loss = 0
        num_instances = 0
        for batch in data_loader:
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.shape[0]
            num_instances += x.shape[0]
        print(f"{e+1}/{num_epochs}: loss = {epoch_loss:.5f}")

    x = np.linspace(0, 1, 100)
    x = x.reshape(1,100,1)
    y = model(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()[0]
    if ax is not None:
        ax.plot(x[0],y, label="model")
        ax.plot(x[0], target_func(x[0]), label="target func")
        ax.legend()
        ax.set_yticks([])
        ax.set_xticks([])
        x, y = data_loader.dataset.tensors
        ax.scatter(x,y, s=10, alpha=0.2)
        ax.set_title(f"n_samples={n}, n_hiddens={num_hidden_units}")
    return y

def decrease_variance():
    num_hidden_units = 3
    num_epochs=100
    lr=0.1
    fig, axes = plt.subplots(4,3,figsize=(19.8, 10.8))
    axes = axes.ravel()
    for i, n in enumerate([6, 10, 100]):
        train(ax=axes[i], n=n, num_epochs=num_epochs, lr=lr, num_hidden_units=num_hidden_units)
        train(ax=axes[i+3], n=n, num_epochs=num_epochs, lr=lr, num_hidden_units=num_hidden_units)
        train(ax=axes[i+6], n=n, num_epochs=num_epochs, lr=lr, num_hidden_units=num_hidden_units)
        y_pred = []
        for _ in range(10):
            y=train(ax=None, n=n, num_epochs=num_epochs, lr=lr, num_hidden_units=num_hidden_units)
            y_pred.append(y)
        y_pred = np.array(y_pred)
        mean_pred = y_pred.mean(axis=0)
        std_pred = y_pred.std(axis=0)
        x = np.linspace(0, 1, 100)
        axes[i+9].plot(x, target_func(x), label="target func")
        axes[i+9].plot(x, mean_pred, label="mean model")
        axes[i+9].set_title(f"n_samples={n}, n_hiddens={num_hidden_units}")
        axes[i+9].plot(x, mean_pred, label="mean model")
        axes[i+9].fill_between(x, (mean_pred-std_pred).reshape(-1), (mean_pred+std_pred).reshape(-1), color="gray", alpha=0.3)
        axes[i+9].legend()
    plt.show()

def bias_variance_tradeoff():
    n = 100
    num_epochs=100
    lr=0.1
    fig, axes = plt.subplots(2,3,figsize=(19.8, 10.8))
    axes = axes.ravel()
    for i, num_hidden_units in enumerate([3, 5, 100]):
        train(ax=axes[i], n=n, num_epochs=num_epochs, lr=lr, num_hidden_units=num_hidden_units)
        y_pred = []
        for _ in range(10):
            y=train(ax=None, n=n, num_epochs=num_epochs, lr=lr, num_hidden_units=num_hidden_units)
            y_pred.append(y)
        y_pred = np.array(y_pred)
        mean_pred = y_pred.mean(axis=0)
        std_pred = y_pred.std(axis=0)
        x = np.linspace(0, 1, 100)
        axes[i+3].plot(x, target_func(x), label="target func")
        axes[i+3].plot(x, mean_pred, label="mean model")
        axes[i+3].set_title(f"n_samples={n}, n_hiddens={num_hidden_units}")
        axes[i+3].plot(x, mean_pred, label="mean model")
        axes[i+3].fill_between(x, (mean_pred-std_pred).reshape(-1), (mean_pred+std_pred).reshape(-1), color="gray", alpha=0.3)
        axes[i+3].legend()
    plt.show()

if __name__ == "__main__":
    functions = [
        decrease_variance,
        bias_variance_tradeoff
    ]
    if len(sys.argv) !=2:
        print("Usage: %s <function id>" % sys.argv[0])
        print()
        print("id | function")
        print("---+"+'-'*20)
        for id, f in enumerate(functions):
            print("%d  | %s" %(id, f.__name__))
        sys.exit()

    id = int(sys.argv[1])
    if(id < 0 or id >= len(functions)) :
        print("Function id %d is invalid (should be in [0, %d])" % (id, len(functions)-1))
        sys.exit()
    functions[id]()
