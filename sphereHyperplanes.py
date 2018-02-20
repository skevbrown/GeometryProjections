# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:18:10 2018

@author: skevb
"""

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

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(azim, polar, r):
    rcos_theta = r * np.cos(polar)
    x = rcos_theta * np.cos(azim)
    y = rcos_theta * np.sin(azim)
    z = r * np.sin(polar)
    return x, y, z

# Vectorize these functions
sph2cartvec = np.vectorize(sph2cart);
cart2sphvec = np.vectorize(cart2sph);



xx1, yy1 = np.meshgrid(np.arange(-3,3.1,0.1),np.arange(-3,3.1,0.1))
#d1 = -np.sum(point1*normal1)
z1 = xx1 * 0

circleAx = np.arange(0, 2*np.pi, 2*np.pi/100)
circlePol = np.array([0]* len(circleAx))
circleRad = np.array([1.0]* len(circleAx))

circleX, circleY, circleZ = sph2cartvec(circleAx,circlePol,circleRad)

npt = 40; npi = npt-1; nptB = npt*2
indUp = 1+1/npi
 
axim =  np.arange(0, indUp, 1/npi) ; axim = axim*np.pi
polar = np.arange(0, indUp, 1/npi) ; polar = polar*np.pi/2
#radius = np.array([1.0] * len(axim))

axisMesh, polMesh = np.meshgrid(axim,polar)
radius = np.array([1.0] * (npt)*(npt))


radMesh = radius
radMesh = radMesh.reshape(npt,npt)
xSphere, ySphere, zSphere = sph2cartvec(axisMesh,polMesh,radMesh)
ySphere2 = -ySphere

zSphere2 = xSphere*0


lineMat = np.zeros(6,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])

        
fig1 = plt.figure(1)
Wire3 = fig1.gca(projection='3d')

# The two blue half spheres
Wire3.plot_wireframe(xSphere,ySphere,zSphere,color=[0.3,0.3,1.0,0.3])
Wire3.plot_wireframe(xSphere,ySphere2,zSphere,color=[0.3,0.3,1.0,0.3])
Wire3.set_xlim([-3,3]); Wire3.set_ylim([-3,3])
Wire3.set_zlim(-1.0,1.0)

# Line from (0,0,1)
lineMat[0] = np.array([0,0,1.0])
lineMat[1] = np.array([np.sqrt(2),0,0])
Wire3.plot(lineMat['xPnts'][0:2],lineMat['yPnts'][0:2],
lineMat['zPnts'][0:2],color='red',linewidth=1.5)

# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter((2/3)*np.sqrt(2),0,1/3,color='blue',s=35)
Wire3.scatter(np.sqrt(2),0,0,color='blue',s=35)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


# Red flat grid plane
Wire3.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=8)

# A circle of Radius 1
Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)


fig2 = plt.figure(2)
Wire4 = fig2.gca(projection='3d')

# The two blue half spheres
Wire4.plot_wireframe(xSphere,ySphere,-zSphere,color=[0.3,0.3,1.0,0.3])
Wire4.plot_wireframe(xSphere,ySphere2,-zSphere,color=[0.3,0.3,1.0,0.3])
Wire4.set_xlim([-3,3]); Wire3.set_ylim([-3,3])
Wire4.set_zlim(-1.0,1.0)

# Circle in Z=0 with radius 1
Wire4.scatter(xSphere,ySphere,zSphere2,color=[1.0,0,0,0.5],s=6)
Wire4.scatter(xSphere,-ySphere,zSphere2,color=[1.0,0,0,0.5],s=6)
Wire4.set_xlim([-3,3]); Wire4.set_ylim([-3,3])
Wire4.set_zlim(-1.0,1.0)

# Line from (0,0,1)
lineMat[2] = np.array([0,0,1.0])
lineMat[3] = np.array([np.sqrt(2)/3,np.sqrt(6)/3,-1/3])
Wire4.plot(lineMat['xPnts'][2:4],lineMat['yPnts'][2:4],
lineMat['zPnts'][2:4],color='blue',linewidth=1.5)

# Line from (0,0,-1)
lineMat[4] = np.array([0,0,-1.0])
lineMat[5] = np.array([np.sqrt(2)/3,np.sqrt(6)/3,1/3])
Wire4.plot(lineMat['xPnts'][4:6],lineMat['yPnts'][4:6],
lineMat['zPnts'][4:6],color='green',linewidth=1.5)

# Point at (2/3 sqrt 2, 0, 1/3)
Wire4.scatter(1/(2*np.sqrt(2)),np.sqrt(3)/(2*np.sqrt(2)),0,color='blue',s=50)
Wire4.scatter(np.sqrt(2)/3,np.sqrt(6)/3,-1/3,color='red',s=50)
Wire4.scatter(np.sqrt(2)/3,np.sqrt(6)/3,1/3,color='red',s=50)
plt.show()