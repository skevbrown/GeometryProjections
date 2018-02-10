# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import PyQt5
import math
import itertools

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    
def check_orthog(pos1,pos2,pos3,matrix):
      mat = np.copy(matrix)
      v1 = np.array(list(mat[pos2])) - np.array(list(mat[pos1]))
      v2 = np.array(list(mat[pos3])) - np.array(list(mat[pos1]))
      npdot = np.dot(v1,v2)
      if npdot == 0:
          print("Orthogonal {}".format(npdot))
      else:
          print("Not Orthogonal {}".format(npdot))


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


bv = np.sin(np.pi/5) # Base value of matrix
bv = 0.578 # Base value of matrix
cubeMat = np.array([[-bv,-bv,-bv],[-bv,bv,-bv],[bv,bv,-bv],[bv,-bv,-bv],[-bv,-bv,-bv],
[-bv,-bv,bv],[-bv,bv,bv],[bv,bv,bv],[bv,-bv,bv],[-bv,-bv,bv], # All the points
[-bv,-bv,-bv],[bv,bv,bv]]) # The line

cubePoints = np.zeros(len(cubeMat),dtype=[('xPnts',np.float64),
('yPnts',np.float64),('zPnts',np.float64)])
cubeRot1 = np.copy(cubePoints)
cubeRot2 = np.copy(cubeRot1)

cubePoints['xPnts'] = cubeMat[:,0]
cubePoints['yPnts'] = cubeMat[:,1]
cubePoints['zPnts'] = cubeMat[:,2]

check_orthog(0,1,3,cubePoints)
lineStart = np.array(list(cubePoints[11])) - np.array(list(cubePoints[10]))
magLS = np.sqrt(lineStart[0]**2+lineStart[1]**2+lineStart[2]**2)
print("Initial Diag length {}".format(magLS))

v = [3, 5, 0]
axis = [0,1,0]
theta = -np.pi/4 

for ii in range(0,len(cubePoints)):
    rowOut = np.dot(rotation_matrix(axis,theta), list(cubePoints[ii]))
    cubeRot1[ii] = rowOut
    print(rowOut ) 
    
check_orthog(0,1,3,cubeRot1)

    
v = [3, 5, 0]
axis = [1,0,0]
theta = (0.2)*np.pi # Eigth/Fortieths ???

for ii in range(0,len(cubeRot1)):
            #print(cubePoints[ii])
    rowOut = np.dot(rotation_matrix(axis,theta), list(cubeRot1[ii]))
    cubeRot2[ii] = rowOut
    print(rowOut ) 
    
check_orthog(0,1,3,cubeRot2)
lineStart = np.array(list(cubeRot2[11])) - np.array(list(cubeRot2[10]))
magLS = np.sqrt(lineStart[0]**2+lineStart[1]**2+lineStart[2]**2)
print("Final Diag length {}".format(magLS))

cubePlot = np.copy(cubeRot2)

plt3d = plt.figure().gca(projection='3d')

plt3d.scatter(cubePoints['xPnts'][0:9],cubePoints['yPnts'][0:9],
cubePoints['zPnts'][0:9],color='red')
plt3d.plot(cubePoints['xPnts'][0:5],cubePoints['yPnts'][0:5],
cubePoints['zPnts'][0:5],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][5:10],cubePoints['yPnts'][5:10],
cubePoints['zPnts'][5:10],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][10:12],cubePoints['yPnts'][10:12],
cubePoints['zPnts'][10:12],color='red',linewidth=1.5)


plt3dRot = plt.figure().gca(projection='3d')

plt3dRot.scatter(cubePlot['xPnts'][0:9],cubePlot['yPnts'][0:9],
cubePlot['zPnts'][0:9],color='red')
plt3dRot.plot(cubePlot['xPnts'][0:5],cubePlot['yPnts'][0:5],
cubePlot['zPnts'][0:5],label='Cube',color='blue',linewidth=0.7)
plt3dRot.plot(cubePlot['xPnts'][5:10],cubePlot['yPnts'][5:10],
cubePlot['zPnts'][5:10],label='Cube',color='blue',linewidth=0.7)
plt3dRot.plot(cubePlot['xPnts'][10:12],cubePlot['yPnts'][10:12],
cubePlot['zPnts'][10:12],color='red',linewidth=1.5)


plt.show()