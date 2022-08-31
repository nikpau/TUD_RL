import matplotlib.pyplot as plt
import numpy as np

iters = np.arange(0,20.1,0.1)
gammas = np.array([0.9,0.8,0.5])
res = np.empty((3,len(iters)))

for idx, gamma in enumerate(gammas):
    res[idx] = list(map(lambda x: gamma**x, iters))

plt.figure()
plt.plot(iters,res[0],lw = 6)
plt.plot(iters,res[1],lw = 6)
plt.plot(iters,res[2],lw = 6)
plt.axis("off")
plt.show()