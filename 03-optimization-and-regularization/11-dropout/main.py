#!/home/malick/miniconda3/envs/pt/bin/python3

import torch
import torch.nn as nn
import utils
import dropout


X = torch.arange(16, dtype=torch.float32).reshape((2, 8))

print("Example of application of dropout:")
print("dropout of 0:", dropout.dropout_layer(X, 0))
print("dropout of 0.5:", dropout.dropout_layer(X, 0.5))
print("dropout of 1:", dropout.dropout_layer(X, 1))
print("\n\n")


hyperparameters = {
    "input_size": 784, "output_size": 10,
    "num_hiddens1": 256, "num_hiddens2": 256,
    "dropout1": 0.5, "dropout2": 0.5
}

learning_rate = 0.1
data = utils.MnistData()
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dropout from scratch")
dropout_mlp_from_scratch = dropout.DropoutMLPScratch(**hyperparameters)

tracks_mlp_from_scratch = utils.train(
    model = dropout_mlp_from_scratch,
    optimizer=torch.optim.SGD(params=dropout_mlp_from_scratch.parameters(), lr=learning_rate),
    criterion=nn.CrossEntropyLoss(), data=data, epochs=num_epochs, device=device
)

utils.plot_loss_and_metric(tracks_mlp_from_scratch)

print("Concise dropout")
dropout_mlp = dropout.DropoutMLP(**hyperparameters)

tracks_mlp = utils.train(
    model = dropout_mlp,
    optimizer=torch.optim.SGD(params=dropout_mlp.parameters(), lr=learning_rate),
    criterion=nn.CrossEntropyLoss(), data=data, epochs=num_epochs, device=device
)

utils.plot_loss_and_metric(tracks_mlp)
