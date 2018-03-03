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



xx1, yy1 = np.meshgrid(np.arange(-2,2.05,0.05),np.arange(-2,2.05,0.05))
#d1 = -np.sum(point1*normal1)
z1 = xx1 * 0

# Set up a Circle for Z=0
circleAx = np.arange(0, 2*np.pi, 2*np.pi/100)
circlePol = np.array([0]* len(circleAx))
circleRad = np.array([1.0]* len(circleAx))

circleX, circleY, circleZ = sph2cartvec(circleAx,circlePol,circleRad)

npt = 8; npi = npt-1; nptB = npt*2
indUp = 1+1/npi

# Make sphere Mesh using Axial an Polar coords, turn to mesh, Then Cart
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

# Shortcut this value
Sq2o3 = np.sqrt(2)/3


fig1 = plt.figure(1)
Wire3 = fig1.gca(projection='3d')

# The two blue half spheres
Wire3.plot_wireframe(xSphere,ySphere,zSphere,color=[0.3,0.3,1.0,0.1])
Wire3.plot_wireframe(xSphere,ySphere2,zSphere,color=[0.3,0.3,1.0,0.1])
Wire3.plot_wireframe(xSphere,ySphere,-zSphere, color=[0.3,0.3,1.0,0.1])
Wire3.plot_wireframe(xSphere,ySphere2,-zSphere,color=[0.3,0.3,1.0,0.1])
Wire3.set_xlim([-2,2]); Wire3.set_ylim([-2,2])
Wire3.set_zlim(-1.0,1.0)

# Line from (0,0,1)
#lineMat[0] = np.array([0,0,1.0])
#lineMat[1] = np.array([np.sqrt(2),0,0])
#Wire3.plot(lineMat['xPnts'][0:2],lineMat['yPnts'][0:2],
#lineMat['zPnts'][0:2],color='red',linewidth=1.5)

# Line for ellipse, Ellipse, and setup Normal Vec
noDiv = 200
zl = -0.505; zu = 0.505; incr =(zu-zl)/noDiv
ellipseLine = np.arange(zl,zu+incr,incr); 
ellipseLine = np.append(ellipseLine,-ellipseLine[0])

#ellPad = np.arange(ellipseLine[-2],ellipseLine[-1],(ellipseLine[-1]-ellipseLine[-2])/50)

#ellipseLine = np.append(ellipseLine,ellPad)
    
lenEL = len(ellipseLine)
ellLiMat = np.zeros(lenEL,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])
ellLiMat['zPnts'] = ellipseLine
ellUpX = np.sqrt( 1 - 0 -zu**2 ) 
ellLowX = -np.sqrt( 1 - 0 -zl**2 )
incr2 = (ellUpX-ellLowX)/lenEL
ellipX = np.arange(ellLowX,ellUpX, incr2)
#ellipX[-1] = -ellipX[0]
ellLiMat['xPnts'] = ellipX

ellNormPhi, ellNormTh, ellNormRad = cart2sph(ellUpX,0,zu)
ellNormal = np.zeros([2,3])
ellNormal[1] = sph2cart(ellNormPhi,ellNormTh + np.pi/2,ellNormRad*1.2)

# Lin e that defines Ellipse
#Wire3.scatter(ellLiMat['xPnts'],ellLiMat['yPnts'],ellLiMat['zPnts'],s=8,
#color=[0.2,0.2,0.9,0.9])

Wire3.plot(ellNormal[0:,0],ellNormal[0:,1],ellNormal[0:,2],
color=[0.3,0.8,0.3,0.8],linewidth=3.5)

ellipse = ellLiMat.copy()
ellipse = np.concatenate((ellipse,ellLiMat),axis=0)
ellipse['yPnts'] = np.sqrt( 1 - ellipse['xPnts']**2 - ellipse['zPnts']**2)
ellipseHalf = int(len(ellipse)/2)
ellipse['yPnts'][ellipseHalf:] = -ellipse['yPnts'][ellipseHalf:]
ellipse['yPnts'][0] = 0; ellipse['yPnts'][ellipseHalf-1] = 0;

Wire3.scatter(ellipse['xPnts'],ellipse['yPnts'],ellipse['zPnts'],s=20,
color=[0.8,0.7,0.4,0.9])

# Red flat grid plane
#Wire3.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=8)

# A circle of Radius 1
Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)


Zeta = np.array( [complex(0.0,0.0)]*len(ellipse))
Zeta.real = ellipse['xPnts']/(1-ellipse['zPnts'])
Zeta.imag = ellipse['yPnts']/(1-ellipse['zPnts'])
Eta = np.array( [complex(0.0,0.0)]*len(ellipse))
Eta.real = ellipse['xPnts']/(1+ellipse['zPnts'])
Eta.imag = -ellipse['yPnts']/(1+ellipse['zPnts'])

 
ZetaCenter = (max(Zeta.real) - min(Zeta.real))/2;
ZetaCenter = max(Zeta.real) - ZetaCenter
EtaCenter = -ZetaCenter

fig2 = plt.figure(2)
ZetaHalf = int(len(Zeta)/2)
plt.plot(Zeta.real[:-ZetaHalf],Zeta.imag[:-ZetaHalf],'b.',linewidth=1.5,markersize=8.5)
plt.plot(Zeta.real[ZetaHalf:],Zeta.imag[ZetaHalf:],  'b.',linewidth=1.5,markersize=8.5)
plt.plot(Eta.real[:-ZetaHalf],Eta.imag[:-ZetaHalf],  'b.',linewidth=1.5,markersize=8.5)
plt.plot(Eta.real[ZetaHalf:],Eta.imag[ZetaHalf:],    'b.',linewidth=1.5,markersize=8.5)


plt.xlim([-2.0,2.0]); plt.ylim([-2.0,2.0])
plt.grid()

plt.plot([ZetaCenter,EtaCenter],[0,0],'r+',markersize=8)

# A circle of Radius 1
plt.plot(circleX,circleY,color='r')
plt.show() 