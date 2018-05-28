import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

filename='results4_699_1'
data = pd.read_csv('{}.csv'.format(filename),header=None)
Z=np.array(data)
X = np.arange(0,data.shape[0], 1)
Y = np.arange(0, data.shape[1], 1)
X2D,Y2D = np.meshgrid(X,Y)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X2D,Y2D, Z,cmap=cm.coolwarm)
plt.savefig('{}.png'.format(filename))
plt.show()