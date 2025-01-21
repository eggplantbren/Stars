import dnest4.classic as dn4
dn4.postprocess()

import corner
import numpy as np
import matplotlib.pyplot as plt

# Set up fonts
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

ndim = 2
posterior_sample = np.loadtxt("posterior_sample.txt")

figure = corner.corner(posterior_sample,
    labels=["$\\mu$", "$\\sigma$"], plot_contours=False,
        plot_density=False, fontsize=14 , hist_kwargs={"color":"blue", "alpha":0.3, "histtype":"stepfilled", "edgecolor":"black","lw":"3"} )

axes = np.array(figure.axes).reshape((ndim, ndim))

for i in range(ndim):
	ax = axes[i,i]
	print( ax )
plt.show()

#plt.savefig("cornerplot.png", dpi=450)

