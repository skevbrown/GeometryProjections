# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import PyQt5

cubeMat = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],
[1,0,1],[1,1,1],[0,1,1], [0,0,0],[0,0,1], [1,0,0],[1,0,1],
[1,1,0],[1,1,1], [0,0,0],[0,1,0], [0,0,0],[1,1,1],
[0,1,1],[1,0,1], [0,1,0],[1,0,0], # Lines in cut planes
[0,0,0],[1/3,1/3,1/3],[1/3,1/3,1/3],[2/3,2/3,2/3]]) # Diag angle points

cubePoints = np.zeros(len(cubeMat),dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])


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
plt3d.plot(cubePoints['xPnts'][17:19],cubePoints['yPnts'][17:19],
cubePoints['zPnts'][17:19],label='Cube',color=(0.7,0.0,0.5,1.0),linewidth=2.0)
plt3d.plot(cubePoints['xPnts'][19:21],cubePoints['yPnts'][19:21],
cubePoints['zPnts'][19:21],label='Cube',color=(0.0,0.0,1.0,1.0),linewidth=2.0)
plt3d.plot(cubePoints['xPnts'][21:23],cubePoints['yPnts'][21:23],
cubePoints['zPnts'][21:23],label='Cube',color=(0.0,0.0,1.0,1.0),linewidth=2.0)

plt3d.scatter(cubePoints['xPnts'],cubePoints['yPnts'],cubePoints['zPnts'],
color='red')

point1 = np.array([0,1,1])
normal1 = np.array([1,1,1])

point2 = np.array([0,0,1])
normal2 = np.array([1,1,1])

#xArray = np.arange(0,1,1/100); yArray = np.arange(0,1,1/100)

xx, yy = np.meshgrid(range(2),range(2))

d1 = -np.sum(point1*normal1) # Dot product
z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1/normal1[2]


d2 = -np.sum(point2*normal2) # Dot product
z2 = (-normal2[0]*xx - normal2[1]*yy - d2)*1/normal2[2]



plt3d.plot_surface(xx,yy,z1,color=(0.3,0.1,0.8,0.5))
plt3d.plot_surface(xx,yy,z2,color=(0.8,0.1,0.3,0.5))
plt3d.set_xlim(0,1.2); plt3d.set_ylim(0,1.2)
plt3d.set_zlim(0,1.2)

plt.show()
