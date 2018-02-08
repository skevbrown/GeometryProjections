# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import PyQt5

cubePoints = np.zeros(17,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])


cubeMat = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],
[1,0,1],[1,1,1],[0,1,1], [0,0,0],[0,0,1], [1,0,0],[1,0,1],
[1,1,0],[1,1,1], [0,0,0],[0,1,0]])

cubePoints['xPnts'] = cubeMat[:,0]
cubePoints['yPnts'] = cubeMat[:,1]
cubePoints['zPnts'] = cubeMat[:,2]


# plot the surface
plt3d = plt.figure().gca(projection='3d')
#plt3d.plot_surface(xx,yy,z1, color=(0.5,0.1,0.9,0.5))
#plt3d.set_xlim(0,1.2); plt3d.set_ylim(0,1.2)
plt3d.plot(cubePoints['xPnts'][0:9],cubePoints['yPnts'][0:9],
cubePoints['zPnts'][0:9],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][9:11],cubePoints['yPnts'][9:11],
cubePoints['zPnts'][9:11],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][11:13],cubePoints['yPnts'][11:13],
cubePoints['zPnts'][11:13],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][13:15],cubePoints['yPnts'][13:15],
cubePoints['zPnts'][13:15],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][15:17],cubePoints['yPnts'][15:17],
cubePoints['zPnts'][15:17],label='Cube',color='blue',linewidth=0.7)


plt3d.scatter(cubePoints['xPnts'],cubePoints['yPnts'],cubePoints['zPnts'],
color='red')

plt.show()
