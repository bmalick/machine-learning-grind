#!/home/malick/miniconda3/envs/pt/bin/python3

import sys
import torch
import matplotlib.pyplot as plt
import convolution

def test_corr2d():
    X = torch.arange(9).reshape((3,3))
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    Y = convolution.corr2d(X=X, K=K)
    print("Input:", X)
    print("Kernel:", K)
    print("Output:", Y)

def test_conv2d_layer():
    X = torch.arange(9).reshape((3,3))
    layer = convolution.Conv2DScratch(kernel_size=2)
    print("Layer kernel:", layer.weight)
    print("Layer bias:", layer.bias)
    print("Layer output:", layer(X))

def test_edge_detection():
    X = torch.ones((6,8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y1 = convolution.corr2d(X=X, K=K)
    Y2 = convolution.corr2d(X=X.t(), K=K)
    fig, axes = plt.subplots(1, 3)
    for t, n, ax in zip([X, Y1, Y2], ["object X", "corr2d(X)", "corr2d(X.T)"], axes):
        ax.imshow(t, cmap="gray")
        ax.set_title(n)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def test_pooling():
    X = torch.arange(9, dtype=torch.float32).reshape((3,3))
    print("X:", X)
    print("max-pool of X with pool_size of 2:", convolution.pool2d(X, (2,2), "max"))
    print("agv-pool of X with pool_size of 2:", convolution.pool2d(X, (2,2), "avg"))

if __name__ == "__main__":
    functions = [
        test_corr2d, test_conv2d_layer,
        test_edge_detection, test_pooling
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
