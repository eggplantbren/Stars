import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("stars_edited.txt", delim_whitespace=True, header=None)
ra = data.iloc[:,2]
dec = data.iloc[:,3]
v = data.iloc[:,6] - np.mean(data.iloc[:,6])

x = (ra - ra.mean())*np.cos(np.mean(dec))
y = (dec - dec.mean())

plt.scatter(x[v > 0], y[v > 0], s=10.0*np.abs(v[v > 0]),
            marker="o", color="red", alpha=0.5)
plt.scatter(x[v <= 0], y[v <= 0], s=10.0*np.abs(v[v <= 0]),
            marker="o", color="blue", alpha=0.5)
plt.axis("equal")
plt.show()

