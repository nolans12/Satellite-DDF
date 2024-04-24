# import PIL
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import os

# # load bluemarble with PIL
# filePath = os.path.dirname(os.path.realpath(__file__))
# bm = PIL.Image.open(filePath + '/blue_marble.jpg')
# # it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept 
# # bm = np.array(bm.resize([d/5 for d in bm.size]))/256.


# # repeat code from one of the examples linked to in the question, except for specifying facecolors:
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# earth_r = 6378.0
# x = earth_r * np.outer(np.cos(u), np.sin(v))
# y = earth_r * np.outer(np.sin(u), np.sin(v))
# z = earth_r * np.outer(np.ones(np.size(u)), np.cos(v))

# ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

# plt.show()

import PIL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# load bluemarble with PIL
filePath = os.path.dirname(os.path.realpath(__file__))
bm = PIL.Image.open(filePath + '/blue_marble.jpg')
# it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept 
bm = np.array(bm.resize([int(d/5) for d in bm.size]))/256.

# coordinates of the image - don't know if this is entirely accurate, but probably close
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

# repeat code from one of the examples linked to in the question, except for specifying facecolors:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

earth_r = 6378.0
x = np.outer(np.cos(lons), np.cos(lats)).T*earth_r
y = np.outer(np.sin(lons), np.cos(lats)).T*earth_r
z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T*earth_r
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

plt.show()