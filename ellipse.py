import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(-5.0, 5.0, 1001),
                   np.linspace(-5.0, 5.0, 1001))
h = x[0, 1] - x[0, 0]


theta = 1.0
xx = x*np.cos(theta) + y*np.sin(theta)
yy = -x*np.sin(theta) + y*np.cos(theta)

q = 0.3
L = 0.7
f = 1.0/(2.0*np.pi*L**2)*np.exp(-0.5*(q*xx**2 + yy**2/q)/L**2)

print(h**2*np.sum(f))

plt.imshow(f, origin="lower")
plt.show()
