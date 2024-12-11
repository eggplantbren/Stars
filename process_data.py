import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("stars_edited.txt", delim_whitespace=True, header=None)
data = data.loc[data.iloc[:, 0] != 5, :]

ra = data.iloc[:,2]
dec = data.iloc[:,3]
v = data.iloc[:,6] - np.mean(data.iloc[:,6])
verr = data.iloc[:,7] - np.mean(data.iloc[:,7])

x = (ra - ra.mean())*np.cos(np.mean(dec))
y = (dec - dec.mean())

# Flip x axis
x = -x

data = np.vstack([x, y, v, verr]).T
np.savetxt("data.txt", data)

plt.scatter(x[v > 0], y[v > 0], s=10.0*np.abs(v[v > 0]),
            marker="o", color="red", alpha=0.5)
plt.scatter(x[v <= 0], y[v <= 0], s=10.0*np.abs(v[v <= 0]),
            marker="o", color="blue", alpha=0.5)
plt.axis("equal")
#plt.gca().invert_xaxis()
plt.show()

