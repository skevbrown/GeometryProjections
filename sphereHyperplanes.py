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



xx1, yy1 = np.meshgrid(np.arange(-10,10),np.arange(-10,10))
#d1 = -np.sum(point1*normal1)
z1 = xx1 * 0

axim = np.arange(0,np.pi,np.pi/30)
polar =  np.arange(0,np.pi/2,(1/30)*np.pi/2)
radius = np.array([1.0] * len(axim))

xSphere, ySphere, zSphere = sph2cartvec(axim,polar,radius)


xx2, yy2 = np.meshgrid(xSphere,ySphere)
z2 = xx2

for i in range(0,len(xx2)):
    for ii in range(0,len(xx2[0])):
        z2[i][ii] = np.sqrt(1-xx2[i][ii]**2 - yy2[i][ii]**2)
        
                
fig1 = plt.figure(1)          # A static 3d plot
Wire3d = fig1.gca(projection='3d') 


# Use this Scaling with range and a generator
#Wire3d.plot_wireframe(xx1*0.01,yy1*0.01,z1,[30,30])

Wire3d.plot_wireframe(xx1,yy1,z1,color=[0.7,0.2,0.3,0.3])
Wire3d.plot_wireframe(xx2,yy2,z2,color=[0.3,0.2,0.7,0.3])


plt.show()