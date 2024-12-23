import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,500)
f = lambda x: -np.log(x)

plt.plot(t, f(t))
plt.title(r"$I(x) = -log P(x)$")
plt.show()
