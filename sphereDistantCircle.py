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
import cmath as cm

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

def zeta(x,y,z):
    re = x/ (1-z)
    im = y/ (1-z)
    comp = np.complex(re,im)
    return comp

def eta(x,y,z):
    re = x/  (1+z)
    im = -y/ (1+z)
    comp = np.complex(re,im)
    return comp

def xyzunit(zReal,zImag):
    denom = 1 + zReal**2 + zImag**2
    x = 2*zReal / denom
    y = 2*zImag / denom
    z = (-1 + zReal**2 + zImag**2) / denom
    xyz = np.array( [x,y,z] )
    return xyz

def xyzEta(zReal,zImag):
    zImag = zImag*(-1)
    sqsum = zReal**2 + zImag**2
    denom = 1 + sqsum
    x = 2*zReal / denom
    y = -2*zImag / denom
    z = (1 - sqsum) / denom
    xyz = np.array( [x,y,z] )
    return xyz

def rect(r, theta):
    """theta in radians

    returns tuple; (float, float); (x,y)
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x,y

def polar(x, y):
    """returns r, theta(radians)
    """
    r, theta = cm.polar( complex(x,y) );

    return r, theta


polarvec = np.vectorize(polar); # Vectorize these func's to operate on vectors
rectvec  = np.vectorize(rect);

# Vectorize these functions
sph2cartvec = np.vectorize(sph2cart);
cart2sphvec = np.vectorize(cart2sph);



xx1, yy1 = np.meshgrid(np.arange(-4,4.1,0.1),np.arange(-4,4.1,0.1))
#d1 = -np.sum(point1*normal1)
z1 = xx1 * 0

circleAx = np.arange(0, 2*np.pi, 2*np.pi/100)
circlePol = np.array([0]* len(circleAx))
circleRad = np.array([1.0]* len(circleAx))

circleX, circleY, circleZ = sph2cartvec(circleAx,circlePol,circleRad)

circleX2 = circleX.copy()+ 2.0
circleY2 = circleY.copy()+ 2.0
circleZ2 = circleZ.copy()

sphCircleX = circleX.copy()
sphCircleY = circleY.copy()
sphCircleZ = circleZ.copy()

for i in range(0,len(sphCircleX)):
    sphCircleX[i], sphCircleY[i], sphCircleZ[i] = xyzunit(circleX2[i],circleY2[i])



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

Sq2o3 = np.sqrt(2)/3
        
fig1 = plt.figure(1)
Wire3 = fig1.gca(projection='3d')

# The two blue half spheres
Wire3.plot_wireframe(xSphere,ySphere,zSphere,color=[0.3,0.3,1.0,0.3])
Wire3.plot_wireframe(xSphere,ySphere2,zSphere,color=[0.3,0.3,1.0,0.3])
Wire3.set_xlim([-4,4]); Wire3.set_ylim([-4,4])
Wire3.set_zlim(-1.0,1.0)

# Line from (0,0,1)
lineMat[0] = np.array([0,0,1.0])
lineMat[1] = np.array([np.sqrt(2),0,0])
Wire3.plot(lineMat['xPnts'][0:2],lineMat['yPnts'][0:2],
lineMat['zPnts'][0:2],color='red',linewidth=1.5)

# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter((2/3)*np.sqrt(2),0,1/3,color='red',s=75)
Wire3.scatter(np.sqrt(2),0,0,color='blue',s=75)
Wire3.scatter(-Sq2o3,np.sqrt(3)*Sq2o3,1/3,color='red',s=75)
Wire3.scatter(-Sq2o3,-np.sqrt(3)*Sq2o3,1/3,color='red',s=75)
Wire3.scatter(np.cos(2*np.pi/3)*np.sqrt(2),np.sin(2*np.pi/3)*np.sqrt(2),0,  color='blue',s=75)
Wire3.scatter(np.cos(-2*np.pi/3)*np.sqrt(2),np.sin(-2*np.pi/3)*np.sqrt(2),0,color='blue',s=75)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


# Red flat grid plane
Wire3.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=8)

# A circle of Radius 1
Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)

# Translated cirlce
Wire3.scatter(circleX2,circleY2,circleZ2,color='blue',s=8)
Wire3.scatter(sphCircleX,sphCircleY,sphCircleZ,color='red',s=8)

fig2 = plt.figure(2)
Wire4 = fig2.gca(projection='3d')

# The two blue half spheres
Wire4.plot_wireframe(xSphere,ySphere,-zSphere, color=[0.3,0.3,1.0,0.3])
Wire4.plot_wireframe(xSphere,ySphere2,-zSphere,color=[0.3,0.3,1.0,0.3])
Wire4.set_xlim([-2,2]); Wire4.set_ylim([-2,2])
Wire4.set_zlim(-1.0,1.0)

# Circle in Z=0 with radius 1
Wire4.scatter(xSphere,ySphere,zSphere2,color=[0.7,0.2,0.6,0.3],s=3)
Wire4.scatter(xSphere,-ySphere,zSphere2,color=[0.6,0.2,0.6,0.3],s=3)


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
Wire4.scatter(1/(2*np.sqrt(2)),np.sqrt(3)/(2*np.sqrt(2)),0,color='red',s=95)
Wire4.scatter(1/(np.sqrt(2))*np.cos(np.pi),(1/np.sqrt(2))*np.sin(np.pi),0,color='red',s=95)
Wire4.scatter(1/(np.sqrt(2))*np.cos(-np.pi/3),(1/np.sqrt(2))*np.sin(-np.pi/3),0,color='red',s=95)
Wire4.scatter(np.sqrt(2)/3,np.sqrt(6)/3,-1/3,color='blue',s=50)
Wire4.scatter(np.sqrt(2)/3,np.sqrt(6)/3,-1/3,color='blue',s=50)
Wire4.scatter(Sq2o3,-np.sqrt(3)*Sq2o3,-1/3,color='blue',s=50)
Wire4.scatter(-2*np.sqrt(2)/3,0,-1/3,color='blue',s=50)

Wire4.scatter(np.sqrt(2)/3,np.sqrt(6)/3,1/3,color='blue',s=50)


plt.show() 