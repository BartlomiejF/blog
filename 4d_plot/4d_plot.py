import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams["font.size"] = 15

dat = pd.read_json("4d_plot_data.json")

xs = dat["n_features"]
ys = dat["n_est"]
zs = dat["mae"]
cs = dat["max_depth"]

xs2 = xs.unique()
ys2 = ys.unique()

xticks = [0.05] + list(np.linspace(0.2, 1, 5))

# normalisation
alpha = ((zs-zs.min())/(zs.max()-zs.min()))

# colour array
rgba_colors = np.zeros((zs.shape[0], 4))
rgba_colors[:, 2] = 1.0
rgba_colors[:, 3] = alpha

# fig, ax = plt.subplots(figsize=(25,15))
# ax.scatter(zs, alpha)
# ax.set(xlabel="mean absolute error",
#           ylabel="normalised")
# plt.tight_layout()
# plt.show()

# power = 5
# fig, ax = plt.subplots(1,2, figsize=(25, 15))
# ax[0].scatter(zs, alpha)
# ax[1].scatter(zs, alpha**power)
# ax[0].set(xlabel="mean absolute error",
#           ylabel="normalised")
# ax[1].set(xlabel="mean absolute error",
#           ylabel=f"normalised^{power}")
# plt.tight_layout()
# plt.show()

rgba_colors[:, 3] = alpha**5
fig, ax = plt.subplots(figsize=(25, 15), subplot_kw={"projection": "3d"})
ax.scatter(xs, ys, cs,
           c=rgba_colors,
           s=alpha**5*100,
           marker=".")
ax.scatter(1, 100, 3, marker="v", c="r", s=100)
ax.set(ylabel="n_estimators",
       xlabel="n_features",
       zlabel="max_depth",
       zticks=cs.unique(),
       zlim=(3, 6),
       xticks=xticks,
       )
ax.set_title("4-th dimension size and transparency\nthe power of power")

plt.tight_layout()
plt.show()
