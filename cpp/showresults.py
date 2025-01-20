import dnest4.classic as dn4
dn4.postprocess()

import corner
import numpy as np
import matplotlib.pyplot as plt

# Set up fonts
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rc("text", usetex=True)

posterior_sample = np.loadtxt("posterior_sample.txt")
corner.corner(posterior_sample,
    labels=["$\\mu_v$", "$\\sigma_v$"], plot_contours=False,
        plot_density=False, fontsize=14)
plt.show()

#plt.savefig("cornerplot.png", dpi=450)

