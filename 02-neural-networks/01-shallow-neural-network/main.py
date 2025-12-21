#!/home/malick/miniconda3/envs/pt/bin/python
import numpy as np
import matplotlib.pyplot as plt
from modules import ShallowNeuralNetwork

import torch
import modules
import trainer

def hidden_unit_linear_regions(hidden_unit, phi, axes, titles, color):
    ax1, ax2, ax3 = axes
    linear_func  = lambda x: hidden_unit.theta0 + hidden_unit.theta1 * x

    for ax in axes:
        ax.plot([0,2], [0,0], color="#abb2b9", linestyle="--")

    ax1.plot([0, 2], [linear_func(0), linear_func(2)], color=color)
    ax1.text(1., -0.85, titles[0], horizontalalignment='center',
             verticalalignment='center', color=color)
    ax2.text(1., -0.85, titles[1], horizontalalignment='center',
             verticalalignment='center', color=color)
    ax3.text(1., -0.85, titles[2], horizontalalignment='center',
             verticalalignment='center', color=color)

    t = np.linspace(0,2,50)
    ax2.plot(t, [hidden_unit(x) for x in t], color=color)
    ax3.plot(t, [phi*hidden_unit(x) for x in t], color=color)



if __name__ == "__main__":
    neural_network = ShallowNeuralNetwork(params=[0.2, -0.8, 0.7, 0.4],
                                          hidden_units_params=[(-0.2, 0.29), (-0.9, 0.8), (1.02, -0.6)])

    for i,p in enumerate(neural_network.params):
        print("phi_%d = %4.1f" % (i+1, p))

    fig, axes = plt.subplots(4,3, figsize=(12, 9), dpi=100)

    for i, (h,color) in enumerate(zip(neural_network.hidden_units, ["#d35400", "#48c9b0", "#1c2833"])):
        print(f"h_{i+1}:", h)
        titles = [rf"$\theta_{{{i+1}0}} + \theta_{{{i+1}1}} x$",
                  rf"$h_{i+1} = a[\theta_{{{i+1}0}} + \theta_{{{i+1}1}} x]$",
                  rf"$\phi_{i+1} h_{i+1}$"]
        hidden_unit_linear_regions(h, phi=neural_network.params[i+1],
                                   axes=[axes[0,i], axes[1,i], axes[2,i]],
                                   titles=titles,
                                   color=color)
    for i,ax in enumerate(axes.ravel()):
        ax.set_aspect('equal')
        ax.set_xlim([0, 2])
        ax.set_ylim([-1, 1])
        ax.set_xticks(ticks=np.arange(0,2.2,0.2), labels=["0.0"]+['']*4+["1.0"]+['']*4+["2.0"])
        if i%3==0:
            ax.set_yticks(ticks=np.arange(-1,1.2,0.2), labels=["-1.0"]+['']*4+["0.0"]+['']*4+["1.0"])
        else:
            ax.set_yticks(ticks=np.arange(-1,1.2,0.2), labels=['']*11)
        if i%3==0: ax.set_ylabel("Output")

    axes[3,0].axis("off")
    axes[3,2].axis("off")

    t = np.linspace(0,2,50)
    axes[3,1].plot(t, [neural_network(x) for x in t])
    axes[3,1].text(1., -0.85, r"$\phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3$", horizontalalignment='center',
             verticalalignment='center', color="#17202a", fontsize=8)

    plt.show()

    data = modules.LinearDataset(w=torch.tensor([1.3, 0.6]),
                        b=torch.tensor([4.9]), n=1000, noise=0.01)
    dataloader = torch.utils.data.DataLoader(dataset=data, shuffle=True, batch_size=16, num_workers=4)
    print("num of batches: ", len(dataloader))
    model = modules.FromScratchLinearModel(num_inputs=2)
    optimizer = trainer.SGD(params=[model.w, model.b], lr=0.01)
    criterion = lambda x,y: ((x-y)**2 / 2).mean()

    print("Params to estimate are:")
    print("w:", dataloader.dataset.w, ", b:", dataloader.dataset.b)
    print("Before training:")
    print("w:", model.w.detach().numpy(), ", b:", model.b.item())
    trainer.train_model(model=model, data=dataloader, criterion=criterion, optimizer=optimizer, epochs=10)
    print("After training")
    print("w:", model.w.detach().numpy(), ", b:", model.b.item())

