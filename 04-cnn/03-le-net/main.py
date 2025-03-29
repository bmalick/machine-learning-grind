#!/home/malick/miniconda3/envs/pt/bin/python3

import utils
import lenet

if __name__ == "__main__":
    model = lenet.LeNet()
    model.apply(lenet.init_cnn)

    utils.train(
        model=model,
        learning_rate=0.1, epochs=10,
        data=utils.MnistData()
    )
