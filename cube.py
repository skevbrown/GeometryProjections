# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import PyQt5

cubePoints = np.zeros(8,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])


cubeMat = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0],[0,0,1],
[1,0,1],[1,1,1]])

cubePoints['xPnts'] = cubeMat[:,0]
cubePoints['yPnts'] = cubeMat[:,1]
cubePoints['zPnts'] = cubeMat[:,2]


# plot the surface
plt3d = plt.figure().gca(projection='3d')
#plt3d.plot_surface(xx,yy,z1, color=(0.5,0.1,0.9,0.5))
#plt3d.set_xlim(0,1.2); plt3d.set_ylim(0,1.2)
plt3d.plot(cubePoints['xPnts'],cubePoints['yPnts'],cubePoints['zPnts'],
label='Cube',color='blue')

plt3d.scatter(cubePoints['xPnts'],cubePoints['yPnts'],cubePoints['zPnts'],
color='red')

plt.show()
