import os
import plotly.express as px
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

n = 10
f = open("error.txt", "r")
a = np.zeros((n,n,n))
data = [float(i) for i in f.readline().split()]

x, y, z = [], [], []
for i in range(n):
    for j in range(n):
        for k in range(n):
            x.append(i)
            y.append(j)
            z.append(k)
            a[i][j][k] = data[(i*n + j) * n + k];

# creating figures
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
  
# setting color bar
color_map = cm.ScalarMappable(cmap=cm.Reds_r)
color_map.set_array(data)


#img = ax.scatter(x, y, z, data, cmap=plt.hot())
#fig.colorbar(img)
img = ax.scatter(x, y, z, color='red')
plt.colorbar(color_map)
  
# displaying plot
plt.show()
