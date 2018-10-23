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
    
def vecRotate(v1,rot,th):
    k = np.matrix([[0,-rot[2],rot[1]],[rot[2],0,-rot[0]],[-rot[1],rot[0],0]])
    k2 = k**2
    ide = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    res = v1*ide + v1*k*np.sin(th) + v1*k2*(1-np.cos(th))
    return res
    
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



xx1, yy1 = np.meshgrid(np.arange(-3.7,4.0,0.3),np.arange(-3.7,4.0,0.3))
#d1 = -np.sum(point1*normal1)
z1 = xx1 * 0

Sq2o3 = np.sqrt(2)/3
phi = 1.61803398875
phiRadius = np.sqrt(1+phi**2)

circleAx = np.arange(0, 2*np.pi, 2*np.pi/100)
circlePol = np.array([0]* len(circleAx))
circleRad = np.array([phiRadius]* len(circleAx))

circleX, circleY, circleZ = sph2cartvec(circleAx,circlePol,circleRad)

circleX2 = circleX.copy()+ 2.0
circleY2 = circleY.copy()+ 2.0
circleZ2 = circleZ.copy()

sphCircleX = circleX.copy()
sphCircleY = circleY.copy()
sphCircleZ = circleZ.copy()

for i in range(0,len(sphCircleX)):
    sphCircleX[i], sphCircleY[i], sphCircleZ[i] = xyzunit(circleX2[i],circleY2[i])



npt = 20; npi = npt-1; nptB = npt*2
indUp = 1+1/npi

axim =  np.arange(0, indUp, 1/npi) ; axim = axim*np.pi
polar = np.arange(0, indUp, 1/npi) ; polar = polar*np.pi/2
#radius = np.array([1.0] * len(axim))

axisMesh, polMesh = np.meshgrid(axim,polar)
radius = np.array([phiRadius] * (npt)*(npt))

radMesh = radius
radMesh = radMesh.reshape(npt,npt)
xSphere, ySphere, zSphere = sph2cartvec(axisMesh,polMesh,radMesh)
ySphere2 = -ySphere

zSphere2 = -zSphere


lineMat = np.zeros(40,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])
lineMat2 = np.zeros(40,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])
lineMat3 = np.zeros(40,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])
lineMat4 = np.zeros(40,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])
phiDist = phi

        
fig1 = plt.figure(1)
Wire3 = fig1.gca(projection='3d')

# The two blue half spheres
Wire3.plot_wireframe(xSphere,ySphere,zSphere,  color=[0.3,0.3,1.0,0.3])
Wire3.plot_wireframe(xSphere,ySphere2,zSphere, color=[0.3,0.3,1.0,0.3])
Wire3.plot_wireframe(xSphere,ySphere,zSphere2, color=[0.3,0.6,0.6,0.3])
Wire3.plot_wireframe(xSphere,ySphere2,zSphere2,color=[0.3,0.6,0.6,0.3])
Wire3.set_xlim([-3,3]); Wire3.set_ylim([-3,3])
Wire3.set_zlim(-1.7,1.7)

# Line from (0,0,1)
#lineMat[0] = np.array([0,0,1.0])
#lineMat[1] = np.array([np.sqrt(2),0,0])
#Wire3.plot(lineMat['xPnts'][0:2],lineMat['yPnts'][0:2],
#lineMat['zPnts'][0:2],color='blue',linewidth=1.5)

# Lines Highlight Triangle
lineMat[0] = np.array([0, -phiDist,1.0])
lineMat[1] = np.array([0, -phiDist,-1.0])
Wire3.plot(lineMat['xPnts'][0:2],lineMat['yPnts'][0:2],
lineMat['zPnts'][0:2],color='red',linewidth=1.5)
lineMat[2] = np.array([0,-phiDist,1.0])
lineMat[3] = np.array([phiDist,-1.0, 0])
Wire3.plot(lineMat['xPnts'][2:4],lineMat['yPnts'][2:4],
lineMat['zPnts'][2:4],color='red',linewidth=1.5)
lineMat[4] = np.array([0, -phiDist,-1.0])
lineMat[5] = np.array([phiDist,-1.0, 0])
Wire3.plot(lineMat['xPnts'][4:6],lineMat['yPnts'][4:6],
lineMat['zPnts'][4:6],color='red',linewidth=1.5)
lineMat[6] = np.array([phiDist, -1.0,0])
lineMat[7] = np.array([phiDist,1.0,0])
Wire3.plot(lineMat['xPnts'][6:8],lineMat['yPnts'][6:8],
lineMat['zPnts'][6:8],color='blue',linewidth=2.5)
lineMat[8] = np.array([phiDist,1.0,0])
lineMat[9] = np.array([1.0,0,phiDist])
Wire3.plot(lineMat['xPnts'][8:10],lineMat['yPnts'][8:10],
lineMat['zPnts'][8:10],color='blue',linewidth=2.5)
lineMat[10] = np.array([1.0,0,phiDist])
lineMat[11] = np.array([phiDist, -1.0,0])
Wire3.plot(lineMat['xPnts'][10:12],lineMat['yPnts'][10:12],
lineMat['zPnts'][10:12],color='blue',linewidth=2.5)

pnt1 = np.array([-1.0,0,np.sqrt(phi)])
# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter(pnt1[0],pnt1[1],pnt1[2],color='red',s=75)
Wire3.scatter( 1.0,0,phiDist,color='red',s=75)
Wire3.scatter(-1.0,0,-phiDist,color='red',s=75)
Wire3.scatter( 1.0,0,-phiDist,color='red',s=75)
Wire3.scatter(0, phiDist,-1.0,color='red',s=75)
Wire3.scatter(0, phiDist,1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,-1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,1.0,color='red',s=75)
Wire3.scatter( phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter( phiDist,  1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist,  1.0, 0,color='red',s=75)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


# Red flat grid plane
#Wire3.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=8)

# A circle of Radius 1 (EQUATOR)
Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)

# Translated cirlce
#Wire3.scatter(circleX2,circleY2,circleZ2,color='blue',s=8)
#Wire3.scatter(sphCircleX,sphCircleY,sphCircleZ,color='blue',s=8)

fig2 = plt.figure(2)
Wire3 = fig2.gca(projection='3d')

# The two blue half spheres
#Wire3.plot_wireframe(xSphere,ySphere,zSphere,  color=[0.3,0.3,1.0,0.3])
#Wire3.plot_wireframe(xSphere,ySphere2,zSphere, color=[0.3,0.3,1.0,0.3])
#Wire3.plot_wireframe(xSphere,ySphere,zSphere2, color=[0.3,0.6,0.6,0.3])
#Wire3.plot_wireframe(xSphere,ySphere2,zSphere2,color=[0.3,0.6,0.6,0.3])

# CLOSED LOOP 
# (phi,-1,0) (1,0,phi) (-1,0,phi) (-phi,1,0) (-1,0,-phi) (1,0,-phi)

# Lines Highlight Triangle
lineMat2[6] = np.array([0,-phiDist,-1.0])
lineMat2[7] = np.array([0,-phiDist,1.0])
Wire3.plot(lineMat2['xPnts'][6:8],lineMat2['yPnts'][6:8],
lineMat2['zPnts'][6:8],color='blue',linewidth=2.5)
lineMat2[8] = lineMat2[6]
lineMat2[9] = np.array([phiDist,-1.0,0])
Wire3.plot(lineMat2['xPnts'][8:10],lineMat2['yPnts'][8:10],
lineMat2['zPnts'][8:10],color='blue',linewidth=2.5)
lineMat2[10] = lineMat2[7]
lineMat2[11] = lineMat2[9]
Wire3.plot(lineMat2['xPnts'][10:12],lineMat2['yPnts'][10:12],
lineMat2['zPnts'][10:12],color='blue',linewidth=2.5)

Wire3.set_xlim([-3,3]); Wire3.set_ylim([-3,3])
Wire3.set_zlim(-1.7,1.7)


vec1 = np.array([lineMat2[6]['xPnts'],lineMat2[6]['yPnts'],lineMat2[6]['zPnts']]);
vec2 = np.array([lineMat2[9]['xPnts'],lineMat2[9]['yPnts'],lineMat2[9]['zPnts']]);
vec3 = np.cross(vec1,vec2)
vec3 = vec3/np.sqrt(vec3[0]**2+vec3[1]**2+vec3[2]**2)
#lineMat[12] = np.array([0,0,0])
#lineMat[13] = np.array([vec3[0],vec3[1],vec3[2]])
#Wire3.plot(lineMat['xPnts'][12:14],lineMat['yPnts'][12:14],
#lineMat['zPnts'][12:14],color='blue',linewidth=2.5)

vec4 = vec1.copy()
#lineMat[14] = np.array([0,0,0])
#lineMat[15] = np.array([vec4[0],vec4[1],vec4[2]])
#Wire3.plot(lineMat['xPnts'][14:16],lineMat['yPnts'][14:16],
#lineMat['zPnts'][14:16],color='red',linewidth=2.5)
 
anInt = 50
for ii in range(1,anInt):
    ang = ii*np.pi/anInt
    vec5 = vecRotate(vec4,vec3,-ang)
    vec5 = np.squeeze(np.asarray(vec5))
    #Wire3.scatter(vec5[0],vec5[1],vec5[2],color='blue',s=10)
    vec6 = vecRotate(vec4,vec3,ang)
    vec6 = np.squeeze(np.asarray(vec6))
    #Wire3.scatter(vec6[0],vec6[1],vec6[2],color='blue',s=10)


lineMat2[16] = np.array([1.0,0,phiDist])
lineMat2[17] = np.array([1.0,0,-phiDist])
Wire3.plot(lineMat2['xPnts'][16:18],lineMat2['yPnts'][16:18],
lineMat2['zPnts'][16:18],color='blue',linewidth=2.5)

lineMat2[18] = lineMat2[16]
lineMat2[19] = np.array([-phiDist,-1.0,0])
Wire3.plot(lineMat2['xPnts'][18:20],lineMat2['yPnts'][18:20],
lineMat2['zPnts'][18:20],color='blue',linewidth=2.5)
lineMat2[20] = lineMat2[17]
lineMat2[21] = lineMat2[19]
Wire3.plot(lineMat2['xPnts'][20:22],lineMat2['yPnts'][20:22],
lineMat2['zPnts'][20:22],color='blue',linewidth=2.5)

lineMat2[22] = np.array([0,phiDist,-1.0])
lineMat2[23] = np.array([0,phiDist,1.0])
Wire3.plot(lineMat2['xPnts'][22:24],lineMat2['yPnts'][22:24],
lineMat2['zPnts'][22:24],color='red',linewidth=2.5)
lineMat2[24] = lineMat2[22]
lineMat2[25] = np.array([-phiDist,1.0,0])
Wire3.plot(lineMat2['xPnts'][24:26],lineMat2['yPnts'][24:26],
lineMat2['zPnts'][24:26],color='red',linewidth=2.5)
lineMat2[26] = lineMat2[23]
lineMat2[27] = np.array([-phiDist,1.0,0])
Wire3.plot(lineMat2['xPnts'][26:28],lineMat2['yPnts'][26:28],
lineMat2['zPnts'][26:28],color='red',linewidth=2.5)
lineMat2[28] = np.array([-1.0,0,phiDist])
lineMat2[29] = np.array([-1.0,0,-phiDist])
Wire3.plot(lineMat2['xPnts'][28:30],lineMat2['yPnts'][28:30],
lineMat2['zPnts'][28:30],color='blue',linewidth=2.5)
lineMat2[30] = lineMat2[28]
lineMat2[31] = np.array([phiDist,1.0,0])
Wire3.plot(lineMat2['xPnts'][30:32],lineMat2['yPnts'][30:32],
lineMat2['zPnts'][30:32],color='red',linewidth=2.5)
lineMat2[32] = lineMat2[29]
lineMat2[33] = lineMat2[31]
Wire3.plot(lineMat2['xPnts'][32:34],lineMat2['yPnts'][32:34],
lineMat2['zPnts'][32:34],color='red',linewidth=2.5)

vec33 = np.array([lineMat2[31]['xPnts'],lineMat2[31]['yPnts'],lineMat2[31]['zPnts']])
vec34 = np.array([lineMat2[28]['xPnts'],lineMat2[28]['yPnts'],lineMat2[28]['zPnts']])
vec35 = vec33 - vec34
vec35 = phiRadius*vec35/np.sqrt(vec35[0]**2+vec35[1]**2+vec35[2]**2)
vec36 = np.array([lineMat2[29]['xPnts'],lineMat2[29]['yPnts'],lineMat2[29]['zPnts']])
vec37 = vec36 - vec34
vec38 = np.cross(vec35,vec37)
vec38 = vec38/np.sqrt([vec38[0]**2+vec38[1]**2+vec38[2]**2])


vec7 = np.array([lineMat2[19]['xPnts'],lineMat2[19]['yPnts'],lineMat2[19]['zPnts']]);
vec8 = np.array([lineMat2[17]['xPnts'],lineMat2[17]['yPnts'],lineMat2[17]['zPnts']]);
vec9 = np.cross(vec7,vec8)
vec9 = vec9/np.sqrt(vec9[0]**2+vec9[1]**2+vec9[2]**2)
vec10 = vec7.copy()

anInt = 50
for ii in range(1,anInt):
    ang = ii*np.pi/anInt
    vec39 = vecRotate(vec35,vec38,-ang)
    vec39 = np.squeeze(np.asarray(vec39))
    Wire3.scatter(vec39[0],vec39[1],vec39[2],color='blue',s=10)
    vec40 = vecRotate(vec35,vec38,ang)
    vec40 = np.squeeze(np.asarray(vec40))
    Wire3.scatter(vec40[0],vec40[1],vec40[2],color='blue',s=10)

# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter(-1.0,0,phiDist,color='red',s=75)
Wire3.scatter( 1.0,0,phiDist,color='red',s=75)
Wire3.scatter(-1.0,0,-phiDist,color='red',s=75)
Wire3.scatter( 1.0,0,-phiDist,color='red',s=75)
Wire3.scatter(0, phiDist,-1.0,color='red',s=75)
Wire3.scatter(0, phiDist,1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,-1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,1.0,color='red',s=75)
Wire3.scatter( phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter( phiDist,  1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist,  1.0, 0,color='red',s=75)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


# Red flat grid plane
#Wire3.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=8)

# A circle of Radius 1 (EQUATOR)
#Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)
circleX3 = circleX.copy(); circleY3 = circleY.copy();
circleZ3 = circleY.copy()
circleY3 = circleY3*0
#Wire3.scatter(circleX3,circleY3,circleZ3,color='red', s=8)

circleX4 = circleX.copy(); circleY4 = circleY.copy();
circleZ4 = circleX.copy()
circleX4 = circleY4*0
Wire3.scatter(circleX4,circleY4,circleZ4,color='blue', s=8)

fig3 = plt.figure(3)
Wire3 = fig3.gca(projection='3d')

# The two blue half spheres
#Wire3.plot_wireframe(xSphere,ySphere,zSphere,  color=[0.3,0.3,1.0,0.3])
#Wire3.plot_wireframe(xSphere,ySphere2,zSphere, color=[0.3,0.3,1.0,0.3])
#Wire3.plot_wireframe(xSphere,ySphere,zSphere2, color=[0.3,0.6,0.6,0.3])
#Wire3.plot_wireframe(xSphere,ySphere2,zSphere2,color=[0.3,0.6,0.6,0.3])
Wire3.set_xlim([-3,3]); Wire3.set_ylim([-3,3])
Wire3.set_zlim(-1.7,1.7)

# Lines Highlight Triangle
lineMat3[0] = np.array([phiDist,  -1.0, 0])
lineMat3[1] = np.array([1.0,  0, phiDist])
Wire3.plot(lineMat3['xPnts'][0:2],lineMat3['yPnts'][0:2],lineMat3['zPnts'][0:2],
color=[0.8,0.1,0.1,0.3],linewidth=1.5)
lineMat3[2] = np.array([1.0, 0, phiDist])
lineMat3[3] = np.array([0, -phiDist, 1.0])
Wire3.plot(lineMat3['xPnts'][2:4],lineMat3['yPnts'][2:4],lineMat3['zPnts'][2:4],
color=[0.8,0.1,0.1,0.3],linewidth=1.5)

lineMat3[6] = np.array([0, -phiDist, 1.0])
lineMat3[7] = np.array([0,  -phiDist,-1.0])
Wire3.plot(lineMat3['xPnts'][6:8],lineMat3['yPnts'][6:8],lineMat3['zPnts'][6:8],
color='blue',linewidth=2.5)
lineMat3[8] = np.array([0, -phiDist, 1.0])
lineMat3[9] = np.array([phiDist, -1.0,0])
Wire3.plot(lineMat3['xPnts'][8:10],lineMat3['yPnts'][8:10],lineMat3['zPnts'][8:10],
color='blue',linewidth=2.5)
lineMat3[10] = np.array([0,  -phiDist,-1.0])
lineMat3[11] = np.array([phiDist, -1.0,0])
Wire3.plot(lineMat3['xPnts'][10:12],lineMat3['yPnts'][10:12],lineMat3['zPnts'][10:12],
color='blue',linewidth=2.5)

lineMat3[12] = np.array([1.0, 0, phiDist])
lineMat3[13] = np.array([phiDist, 1.0,0])
Wire3.plot(lineMat3['xPnts'][12:14],lineMat3['yPnts'][12:14],lineMat3['zPnts'][12:14],
color=[0.1,0.1,0.8,0.3],linewidth=2.5)
lineMat3[14] = np.array([phiDist, 1.0, 0])
lineMat3[15] = np.array([phiDist, -1.0,0])
Wire3.plot(lineMat3['xPnts'][14:16],lineMat3['yPnts'][14:16],lineMat3['zPnts'][14:16],
color=[0.1,0.1,0.8,0.3],linewidth=2.5)
lineMat3[16] = np.array([phiDist, 1.0, 0])
lineMat3[17] = np.array([1.0, 0,-phiDist])
Wire3.plot(lineMat3['xPnts'][16:18],lineMat3['yPnts'][16:18],lineMat3['zPnts'][16:18],
color=[0.1,0.1,0.8,0.3],linewidth=2.5)
lineMat3[18] = np.array([phiDist, -1.0, 0])
lineMat3[19] = np.array([1.0, 0,-phiDist])
Wire3.plot(lineMat3['xPnts'][18:20],lineMat3['yPnts'][18:20],lineMat3['zPnts'][18:20],
color=[0.8,0.1,0.1,0.3],linewidth=2.5)
lineMat3[20] = np.array([1.0, 0, -phiDist])
lineMat3[21] = np.array([0, -phiDist,-1.0])
Wire3.plot(lineMat3['xPnts'][20:22],lineMat3['yPnts'][20:22],lineMat3['zPnts'][20:22],
color=[0.8,0.1,0.1,0.3],linewidth=2.5)
lineMat3[22] = np.array([0, -phiDist, 1.0])
lineMat3[23] = np.array([-phiDist,-1.0 ,0])
Wire3.plot(lineMat3['xPnts'][22:24],lineMat3['yPnts'][22:24],lineMat3['zPnts'][22:24],
color=[0.8,0.1,0.1,0.3],linewidth=2.5)
lineMat3[24] = np.array([0, -phiDist, -1.0])
lineMat3[25] = np.array([-phiDist,-1.0 ,0])
Wire3.plot(lineMat3['xPnts'][24:26],lineMat3['yPnts'][24:26],lineMat3['zPnts'][24:26],
color=[0.8,0.1,0.1,0.3],linewidth=2.5)

pnt1 = np.array([-1.0,0,np.sqrt(phi)])
# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter(pnt1[0],pnt1[1],pnt1[2],color='red',s=75)
Wire3.scatter( 1.0,0,phiDist,color='red',s=75)
Wire3.scatter(-1.0,0,-phiDist,color='red',s=75)
Wire3.scatter( 1.0,0,-phiDist,color='red',s=75)
Wire3.scatter(0, phiDist,-1.0,color='red',s=75)
Wire3.scatter(0, phiDist,1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,-1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,1.0,color='red',s=75)
Wire3.scatter( phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter( phiDist,  1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist,  1.0, 0,color='red',s=75)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])

# A circle of Radius 1 (EQUATOR)
Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)

fig4 = plt.figure(4)
Wire3 = fig4.gca(projection='3d')

# The two blue half spheres
Wire3.set_xlim([-3,3]); Wire3.set_ylim([-3,3])
Wire3.set_zlim(-1.7,1.7)

# Lines Highlight Triangle
lineMat4[0] = np.array([phiDist, -1.0,0])
lineMat4[1] = np.array([1.0,  0, phiDist])
Wire3.plot(lineMat['xPnts'][0:2],lineMat['yPnts'][0:2],
lineMat4['zPnts'][0:2],color='red',linewidth=1.5)
lineMat4[2] = np.array([phiDist, -1.0,0])
lineMat4[3] = np.array([1.0, 0, -phiDist])
Wire3.plot(lineMat['xPnts'][2:4],lineMat['yPnts'][2:4],
lineMat4['zPnts'][2:4],color='red',linewidth=1.5)
lineMat4[4] = np.array([phiDist, -1.0,0])
lineMat4[5] = np.array([phiDist,  1.0, 0])
Wire3.plot(lineMat4['xPnts'][4:6],lineMat4['yPnts'][4:6],
lineMat4['zPnts'][4:6],color='red',linewidth=1.5)

lineMat4[6] = np.array([0, -phiDist, 1.0])
lineMat4[7] = np.array([0,  -phiDist,-1.0])
Wire3.plot(lineMat4['xPnts'][6:8],lineMat4['yPnts'][6:8],
lineMat4['zPnts'][6:8],color='blue',linewidth=2.5)
lineMat4[8] = np.array([0, -phiDist, 1.0])
lineMat4[9] = np.array([phiDist, -1.0,0])
Wire3.plot(lineMat['xPnts'][8:10],lineMat['yPnts'][8:10],
lineMat4['zPnts'][8:10],color='blue',linewidth=2.5)
lineMat4[10] = np.array([0,  -phiDist,-1.0])
lineMat4[11] = np.array([phiDist, -1.0,0])
Wire3.plot(lineMat4['xPnts'][10:12],lineMat4['yPnts'][10:12],
lineMat4['zPnts'][10:12],color='blue',linewidth=2.5)

lineMat4[12] = np.array([phiDist,  1.0, 0])
lineMat4[13] = np.array([1.0,  0, phiDist])
Wire3.plot(lineMat4['xPnts'][12:14],lineMat4['yPnts'][12:14],
lineMat4['zPnts'][12:14],color='red',linewidth=1.5)
lineMat4[14] = np.array([phiDist,  1.0, 0])
lineMat4[15] = np.array([1.0,  0, -phiDist])
Wire3.plot(lineMat4['xPnts'][14:16],lineMat4['yPnts'][14:16],
lineMat4['zPnts'][14:16],color='red',linewidth=1.5)
lineMat4[16] = np.array([1.0,  0, -phiDist])
lineMat4[17] = np.array([0,  -phiDist, -1.0])
Wire3.plot(lineMat4['xPnts'][16:18],lineMat4['yPnts'][16:18],
lineMat4['zPnts'][16:18],color='blue',linewidth=1.5)
lineMat4[18] = np.array([1.0,  0, phiDist])
lineMat4[19] = np.array([0,  -phiDist, 1.0])
Wire3.plot(lineMat4['xPnts'][18:20],lineMat4['yPnts'][18:20],
lineMat4['zPnts'][18:20],color='blue',linewidth=1.5)

vec13 = np.array([lineMat4[0]['xPnts'],lineMat4[0]['yPnts'],lineMat4[0]['zPnts']]);
vec14 = np.array([lineMat4[1]['xPnts'],lineMat4[1]['yPnts'],lineMat4[1]['zPnts']]);
vec15 = np.cross(vec13,vec14)
vec15 = vec15/np.sqrt(vec15[0]**2+vec15[1]**2+vec15[2]**2)

anInt = 30
for ii in range(1,anInt):
    ang = ii*(2/3)*np.pi/anInt
    vec16 = vecRotate(vec13,vec15,-ang)
    vec16 = np.squeeze(np.asarray(vec16))
    Wire3.scatter(vec16[0],vec16[1],vec16[2],color='blue',s=8)
    vec17 = vecRotate(vec13,vec15,ang)
    vec17 = np.squeeze(np.asarray(vec17))
    Wire3.scatter(vec17[0],vec17[1],vec17[2],color='blue',s=8)

vec18 = np.array([lineMat4[0]['xPnts'],lineMat4[0]['yPnts'],lineMat4[0]['zPnts']]);
vec19 = np.array([lineMat4[6]['xPnts'],lineMat4[6]['yPnts'],lineMat4[6]['zPnts']]);
vec20 = np.cross(vec18,vec19)
vec20 = vec20/np.sqrt(vec20[0]**2+vec20[1]**2+vec20[2]**2)

for ii in range(1,anInt):
    ang = ii*(2/3)*np.pi/anInt
    vec21 = vecRotate(vec18,vec20,-ang)
    vec21 = np.squeeze(np.asarray(vec21))
    Wire3.scatter(vec21[0],vec21[1],vec21[2],color='blue',s=8)
    vec22 = vecRotate(vec18,vec20,ang)
    vec22 = np.squeeze(np.asarray(vec22))
    Wire3.scatter(vec22[0],vec22[1],vec22[2],color='blue',s=8)

vec23 = np.array([lineMat4[0]['xPnts'],lineMat4[0]['yPnts'],lineMat4[0]['zPnts']]);
vec24 = np.array([lineMat4[3]['xPnts'],lineMat4[3]['yPnts'],lineMat4[3]['zPnts']]);
vec25 = np.cross(vec23,vec24)
vec25 = vec25/np.sqrt(vec25[0]**2+vec25[1]**2+vec25[2]**2)

for ii in range(1,anInt):
    ang = ii*(2/3)*np.pi/anInt
    vec26 = vecRotate(vec23,vec25,-ang)
    vec26 = np.squeeze(np.asarray(vec26))
    Wire3.scatter(vec26[0],vec26[1],vec26[2],color='blue',s=8)
    vec27 = vecRotate(vec23,vec25,ang)
    vec27 = np.squeeze(np.asarray(vec27))
    Wire3.scatter(vec27[0],vec27[1],vec27[2],color='blue',s=8)

vec28 = np.array([lineMat4[0]['xPnts'],lineMat4[0]['yPnts'],lineMat4[0]['zPnts']]);
vec29 = np.array([lineMat4[7]['xPnts'],lineMat4[7]['yPnts'],lineMat4[7]['zPnts']]);
vec30 = np.cross(vec28,vec29)
vec30 = vec30/np.sqrt(vec30[0]**2+vec30[1]**2+vec30[2]**2)

for ii in range(1,anInt):
    ang = ii*(2/3)*np.pi/anInt
    vec31 = vecRotate(vec28,vec30,-ang)
    vec31 = np.squeeze(np.asarray(vec31))
    Wire3.scatter(vec31[0],vec31[1],vec31[2],color='blue',s=8)
    vec32 = vecRotate(vec28,vec30,ang)
    vec32 = np.squeeze(np.asarray(vec32))
    Wire3.scatter(vec32[0],vec32[1],vec32[2],color='blue',s=8)

pnt1 = np.array([-1.0,0,phiDist])

# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter(pnt1[0],pnt1[1],pnt1[2],color='red',s=75)
Wire3.scatter( 1.0,0,phiDist,color='red',s=75)
Wire3.scatter(-1.0,0,-phiDist,color='red',s=75)
Wire3.scatter( 1.0,0,-phiDist,color='red',s=75)
Wire3.scatter(0, phiDist,-1.0,color='red',s=75)
Wire3.scatter(0, phiDist,1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,-1.0,color='red',s=75)
Wire3.scatter(0,-phiDist,1.0,color='red',s=75)
Wire3.scatter( phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter( phiDist,  1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist, -1.0, 0,color='red',s=75)
Wire3.scatter(-phiDist,  1.0, 0,color='red',s=75)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])

# A circle of Radius 1 (EQUATOR)
Wire3.scatter(circleX,circleY,circleZ,color='red',s=8)

plt.show() 