import numpy as np

def create_dataset(a: float, b: float):
    # y = ax + b + epsilon
    # x = np.random.randn(100, 1)
    def generate_float(): return np.random.random()
    x = np.random.normal(generate_float(), generate_float(), (100,1))
    noise = np.random.normal(generate_float(), generate_float(), (100, 1))
    y = a*x + b + noise
    return x, y



def decision_boundary(ax, xbounds, ybounds, colors, model, N: int):
    dx = (xbounds[1] - xbounds[0])/float(N)
    dy = (ybounds[1] - ybounds[0])/float(N)

    xx, yy = np.meshgrid(np.arange(xbounds[0]-0.1, xbounds[1]+0.1, dx),
                         np.arange(ybounds[0]-0.1, ybounds[1]+0.1, dy))
    y_hat = model.predict(np.c_[xx.ravel(), yy.ravel()])
    y_hat = (y_hat > 0.5).astype(int)
    y_hat = y_hat.reshape(xx.shape)
    ax.contourf(xx, yy, y_hat, alpha=0.4,
                levels=[0, 0.5, 1.0], colors=colors)
