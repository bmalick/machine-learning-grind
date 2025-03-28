#!/home/malick/miniconda3/envs/pt/bin/python3

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import utils

fully_connected_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784, out_features=285),
    nn.Linear(in_features=285, out_features=135),
    nn.Linear(in_features=135, out_features=60),
    nn.Linear(in_features=60, out_features=10),
)
conv1d_model = nn.Sequential(
    nn.Flatten(start_dim=2),
    nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3, stride=2, padding=0),
    nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding=0),
    nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(in_features=1455, out_features=10),
)


print("Fully connected model:", utils.compute_number_of_parameters(fully_connected_model), "parameters\n", fully_connected_model)
fcn_tracks = utils.train(model=fully_connected_model,
    optimizer=torch.optim.SGD(lr=0.01, params=fully_connected_model.parameters()),
    criterion=nn.CrossEntropyLoss(),
    data=utils.MnistData(),
    epochs=200,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

print("1D convolutionnal model:", utils.compute_number_of_parameters(conv1d_model), "parameters\n", conv1d_model)
cnn_tracks = utils.train(model=conv1d_model,
    optimizer=torch.optim.SGD(lr=0.01, params=conv1d_model.parameters()),
    criterion=nn.CrossEntropyLoss(),
    data=utils.MnistData(),
    epochs=200,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
axes = axes.ravel()
axes[0].set_title("Fully connected network")
axes[1].set_title("Convolutional network")
axes[2].set_title("FCN and CNN train loss")
axes[3].set_title("FCN and CNN eval loss")
for l,n in zip(fcn_tracks[:2], ["train", "eval"]): axes[0].plot(l, label=n)
for l,n in zip(cnn_tracks[:2], ["train", "eval"]): axes[1].plot(l, label=n)
axes[2].plot(fcn_tracks[0], label="FCN")
axes[2].plot(cnn_tracks[0], label="CNN")
axes[3].plot(fcn_tracks[1], label="FCN")
axes[3].plot(cnn_tracks[1], label="CNN")
for ax in axes: ax.legend()
plt.savefig("cnn-vs-fcn-loss.png")
plt.close()
fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
axes = axes.ravel()
axes[0].set_title("Fully connected network")
axes[1].set_title("Convolutional network")
axes[2].set_title("FCN and CNN train acc")
axes[3].set_title("FCN and CNN eval acc")
for l,n in zip(fcn_tracks[2:], ["train", "eval"]): axes[0].plot(l, label=n)
for l,n in zip(cnn_tracks[2:], ["train", "eval"]): axes[1].plot(l, label=n)
axes[2].plot(fcn_tracks[2], label="FCN")
axes[2].plot(cnn_tracks[2], label="CNN")
axes[3].plot(fcn_tracks[3], label="FCN")
axes[3].plot(cnn_tracks[3], label="CNN")
for ax in axes: ax.legend()
plt.savefig("cnn-vs-fcn-acc.png")
# plt.show()
plt.close()
