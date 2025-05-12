#!/home/malick/miniconda3/envs/pt/bin/python3

import sys
import matplotlib.pyplot as plt

import utils



def test1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    utils.plot_pdfs(n0=1, n1=4, alpha=1, beta=1, ax=axes[0])
    utils.plot_pdfs(n0=1, n1=4, alpha=2, beta=2, ax=axes[1])
    plt.show()


def test2():
    utils.bayesian_inference(alpha=1, beta=1, num_tosses=100)

def test3():
    utils.bayesian_inference(alpha=2, beta=2, num_tosses=100)

if __name__ == "__main__":
    functions = [test1, test2, test3]
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
