# -*- coding: utf-8 -*-a
"""
Created on Wed Mar 21 11:50:22 2018

@author: skevb
"""

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



xx1, yy1 = np.meshgrid(np.arange(-2.5,2.54,0.04),np.arange(-2,2.54,0.04))
#d1 = -np.sum(point1*normal1)
z1 = xx1 * 0

circleAx = np.arange(0, 2*np.pi, 2*np.pi/100)
circlePol = np.array([0]* len(circleAx))
circleRad = np.array([1.0]* len(circleAx))

circleX, circleY, circleZ = sph2cartvec(circleAx,circlePol,circleRad)

circleAx2 = np.arange(0, 2*np.pi, 2*np.pi/100)
circlePol2 = np.array([0]* len(circleAx))
circleRad2 = np.array([np.sqrt(2)]* len(circleAx))

circleX2, circleY2, circleZ2 = sph2cartvec(circleAx2,circlePol2,circleRad2)

npt = 20; npi = npt-1; nptB = npt*2
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
Wire3.set_xlim([-6.0,6.0]); Wire3.set_ylim([-6.0,6.0])
Wire3.set_zlim(-1.0,1.0)


lineMat2 = lineMat.copy()
lineMat2[0] = np.array([0,0,-1.0])
lineMat2[1] = np.array([0,0,1.0])
Wire3.plot(lineMat2['xPnts'][0:2],lineMat2['yPnts'][0:2],
lineMat2['zPnts'][0:2],color='blue',linewidth=2.0)
Wire3.scatter(0,0,-1.0,color='blue',s=50)
Wire3.scatter(0,0,1.0,color='blue',s=50)
Wire3.scatter(1/4,0,0,color='white',edgecolor='red',s=75)

# Point at (2/3 sqrt 2, 0, 1/3)
Wire3.scatter(np.sqrt(2),0,0,color='blue',s=75)

Wire3.scatter(2.0,0,0,color='red',s=50)
Wire3.scatter(4.0,0,0,color='red',s=50)
#Wire3.scatter(0,0,-1.0,color='red',s=75)

Wire3.scatter(np.cos(2*np.pi/3)*np.sqrt(2),np.sin(2*np.pi/3)*np.sqrt(2),0,  color='blue',s=75)
Wire3.scatter(np.cos(-2*np.pi/3)*np.sqrt(2),np.sin(-2*np.pi/3)*np.sqrt(2),0,color='blue',s=75)


Wire3.scatter(1,1, 0,color='red',s=50)
Wire3.scatter(1,-1,0,color='red',s=50)
complineMat = lineMat.copy()
complineMat[0] = np.array([0,0,0]); complineMat[1] = np.array([2,3,0])
Wire3.plot(complineMat['xPnts'][0:2],complineMat['yPnts'][0:2],
complineMat['zPnts'][0:2],color='blue',linewidth=8.0)
conjlineMat = lineMat.copy()
conjlineMat[0] = np.array([0,0,0]); conjlineMat[1] = np.array([1,-1,0])
Wire3.plot(conjlineMat['xPnts'][0:2],conjlineMat['yPnts'][0:2],
conjlineMat['zPnts'][0:2],color='red',linewidth=2.0)
lineMat6 = lineMat.copy()
lineMat6[0] = np.array([1.0,0,0]); lineMat6[1] = np.array([5.0,0,0])
Wire3.plot(lineMat6['xPnts'][0:2],lineMat6['yPnts'][0:2],
lineMat6['zPnts'][0:2],color='blue',linewidth=1.5)
radlineMat = lineMat.copy()
radlineMat[0] = np.array([0,0,1.0]); radlineMat[1] = np.array([4.0,0,0])
Wire3.plot(radlineMat['xPnts'][0:2],radlineMat['yPnts'][0:2],
radlineMat['zPnts'][0:2],color='blue',linewidth=1.5)
xyzP6 = xyzunit(4,0)
Wire3.scatter(xyzP6[0],xyzP6[1],xyzP6[2],color='black',s=50)
etalineMat = lineMat.copy()
etalineMat[0] = np.array([0,0,-1.0]); etalineMat[1] = np.array([xyzP6[0],xyzP6[1],xyzP6[2]])
Wire3.plot(etalineMat['xPnts'][0:2],etalineMat['yPnts'][0:2],
etalineMat['zPnts'][0:2],color='blue',linewidth=1.5)
     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


# Red flat grid plane
Wire3.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=8)

# A circle of Radius 1
Wire3.plot(circleX,circleY,circleZ,color='red',linewidth=2.5)

Wire3.plot(circleX2,circleY2,circleZ2,color='blue',linewidth=0.7)

fig2 = plt.figure(2)
Wire4 = fig2.gca(projection='3d')

# The two blue half spheres
Wire4.plot_wireframe(xSphere,ySphere,zSphere,color=[0.3,0.3,1.0,0.3])
Wire4.plot_wireframe(xSphere,ySphere2,zSphere,color=[0.3,0.3,1.0,0.3])
Wire4.set_xlim([-18.0,18.0]); Wire4.set_ylim([-18.0,18.0])
Wire4.set_zlim(-1.5,1.5)


lineMat7 = lineMat.copy()
lineMat7[0] = np.array([0,0,-1.0])
lineMat7[1] = np.array([0,0,1.0])
Wire4.plot(lineMat7['xPnts'][0:2],lineMat7['yPnts'][0:2],
lineMat7['zPnts'][0:2],color='blue',linewidth=2.0)

# Point at (2/3 sqrt 2, 0, 1/3)
Wire4.scatter(np.sqrt(2),0,0,color='blue',s=75)

Wire4.scatter(4.0,0,0,color=[0.1,0.1,0.1,1.0],s=50)

Wire4.scatter(np.cos(2*np.pi/3)*np.sqrt(2),np.sin(2*np.pi/3)*np.sqrt(2),0,  color='blue',s=75)
Wire4.scatter(np.cos(-2*np.pi/3)*np.sqrt(2),np.sin(-2*np.pi/3)*np.sqrt(2),0,color='blue',s=75)
Wire4.scatter(16,0,0,color='red',s=50)

Wire4.scatter(2*np.sqrt(2),2*np.sqrt(2), 0,color='red',s=50)
Wire4.scatter(2*np.sqrt(2),-2*np.sqrt(2),0,color='red',s=50)
lineMat8 = lineMat.copy()
lineMat8[0] = np.array([0,0,0]); lineMat8[1] = np.array([2*np.sqrt(2),2*np.sqrt(2),0])
Wire4.plot(lineMat8['xPnts'][0:2],lineMat8['yPnts'][0:2],
lineMat8['zPnts'][0:2],color='red',linewidth=2.0)
lineMat9 = lineMat.copy()
lineMat9[0] = np.array([0,0,0]); lineMat9[1] = np.array([2*np.sqrt(2),-2*np.sqrt(2),0])
Wire4.plot(lineMat9['xPnts'][0:2],lineMat9['yPnts'][0:2],
lineMat9['zPnts'][0:2],color='red',linewidth=2.0)
lineMat10 = lineMat.copy()
lineMat10[0] = np.array([0,0,0]); lineMat10[1] = np.array([16.0,0,0])
Wire4.plot(lineMat10['xPnts'][0:2],lineMat10['yPnts'][0:2],
lineMat10['zPnts'][0:2],color='blue',linewidth=1.5)
lineMat11 = lineMat.copy()
lineMat11[0] = np.array([0,0,1.0]); lineMat11[1] = np.array([16.0,0,0])
Wire4.plot(lineMat11['xPnts'][0:2],lineMat11['yPnts'][0:2],
lineMat11['zPnts'][0:2],color='blue',linewidth=1.5)
     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


# Red flat grid plane
Wire4.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=1)

# A circle of Radius 1
Wire4.plot(circleX,circleY,circleZ,color='red',linewidth=2.5)

Wire4.plot(circleX2,circleY2,circleZ2,color='blue',linewidth=0.7)


fig3 = plt.figure(3)
Wire5 = fig3.gca(projection='3d')

# The two blue half spheres
Wire5.plot_wireframe(xSphere,ySphere,zSphere,color=[0.3,0.3,1.0,0.3])
Wire5.plot_wireframe(xSphere,ySphere2,zSphere,color=[0.3,0.3,1.0,0.3])
Wire5.set_xlim([-2.0,2.0]); Wire5.set_ylim([-2.0,2.0])
Wire5.set_zlim(-1.2,1.2)


lineMat12 = lineMat.copy()
lineMat12[0] = np.array([0,0,1.0])

zetaOne3 = zeta(np.sqrt(2)/3,np.sqrt(6)/3,1/3)
etaOne3 = eta(2*np.sqrt(2)/3,0,-1/3)

xyzP5 = xyzEta(etaOne3.real,-etaOne3.imag)
#xyzP5b = xyzP5
xyzP5b = np.array( [etaOne3.real,etaOne3.imag,0] )
#xyzP5b[2] = xyzP5b[2]*1.0
#xyzP5b = list( sph2cart(xyzP5b[0],xyzP5b[1],xyzP5b[2]) )
#xyzP5b = np.array(xyzP5b)

# Point at (2/3 sqrt 2, 0, 1/3)
#Wire5.scatter(np.sqrt(2),0,0,color='blue',s=75)

#Wire5.scatter(xyzP5[0],xyzP5[1],xyzP5[2],color='red',s=150)

Wire5.scatter(-np.sqrt(2)/3,np.sqrt(6)/3,-1/3,  color='blue',s=75)
Wire5.scatter(-np.sqrt(2)/3,-np.sqrt(6)/3,-1/3,  color='blue',s=75)
Wire5.scatter(2*np.sqrt(2)/3,0,-1/3,  color='blue',s=75)#Wire5.scatter(np.cos(-2*np.pi/3)*np.sqrt(2),np.sin(-2*np.pi/3)*np.sqrt(2),0,color='blue',s=75)


#Wire5.scatter(1/np.sqrt(2),1/np.sqrt(2), 0,color='red',s=50)
#Wire5.scatter(1/np.sqrt(2),-1/np.sqrt(2),0,color='red',s=50)
lineMat13 = lineMat.copy()
lineMat13[0] = np.array([0,0,-1.0]); 
lineMat13[1] = np.array([xyzP5b[0],xyzP5b[1],xyzP5b[2]])
Wire5.plot(lineMat13['xPnts'][0:2],lineMat13['yPnts'][0:2],
lineMat13['zPnts'][0:2],color='red',linewidth=1.0)



lineMat14 = lineMat.copy()
lineMat14[0] = np.array([0,0,1.0]); 
lineMat14[1] = np.array([2*np.sqrt(2)/3,0,-1/3])
Wire5.plot(lineMat14['xPnts'][0:2],lineMat14['yPnts'][0:2],
lineMat14['zPnts'][0:2],color='red',linewidth=1.0)
Wire5.scatter(1/np.sqrt(2),0, 0,color='red',s=50)
Wire5.scatter(etaOne3.real,-etaOne3.imag,  0,color='red',s=50)

     # Use this Scaling with range and a generator
     #Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])


centAxMat = lineMat.copy()
centAxMat[0] = np.array([0,0,-1.0]); 
centAxMat[1] = np.array([0,0,1.0])
Wire5.plot(centAxMat['xPnts'][0:2],centAxMat['yPnts'][0:2],
centAxMat['zPnts'][0:2],color='blue',linewidth=2.0)
Wire5.scatter(0,0,1.0, color='blue',s=75)
Wire5.scatter(0,0,-1.0,color='blue',s=75)

xAxMat = lineMat.copy()
xAxMat[0] = np.array([-1.0,0,0]); 
xAxMat[1] = np.array([1.0,0,0])
Wire5.plot(xAxMat['xPnts'][0:2],xAxMat['yPnts'][0:2],
xAxMat['zPnts'][0:2],color='blue',linewidth=2.5)

yAxMat = lineMat.copy()
yAxMat[0] = np.array([0,-1.0,0]); 
yAxMat[1] = np.array([0,1.0,0])
Wire5.plot(yAxMat['xPnts'][0:2],yAxMat['yPnts'][0:2],
yAxMat['zPnts'][0:2],color='black',linewidth=2.5)

# Red flat grid plane
Wire5.scatter(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3],s=3)

# A circle of Radius 1
Wire5.plot(circleX,circleY,circleZ,color='red',linewidth=2.5)

#Wire5.plot(circleX2,circleY2,circleZ2,color='blue',linewidth=0.7)


plt.show() 