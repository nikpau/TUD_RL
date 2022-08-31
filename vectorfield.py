# Vector field using quiver
import matplotlib.pyplot as plt

import numpy as np 

K = 0.01

x,y = np.meshgrid(np.arange(-100,110,20), np.arange(0,210,20))

angles = np.arctan(K*x)

u = -np.sin(angles)
v = np.cos(angles)

plt.figure(figsize=(5,5))
plt.quiver(x,y,u,v, scale=15, headwidth=4,width = 0.004, color = "#005f73")
plt.axvline(x=0.0, color= "#9b2226", lw=2)
#plt.savefig("vectorfield.svg")
plt.show()