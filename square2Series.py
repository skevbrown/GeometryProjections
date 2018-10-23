# -*- coding: utf-8 -*-

"""
=====
Decay
=====

This example showcases a sinusoidal decay animation.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from random import *

import cmath as cm
import math

def data_gen(t=0):    # Generates a new data vector each time its called
    cnt = 0
    while cnt < 310:  # This sets the maximum number of points plotted
        cnt += 1
        t += 0.01
        yield np.cos(7*t)*np.cos(t), np.cos(7*t)*np.sin(t)


def init():
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-1.2, 1.2)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

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
          
def mag3D(vector3):
    x = vector3[0]
    y = vector3[1]
    z = vector3[2]
    mag = np.sqrt(x**2 + y**2 + z**2)
    return mag

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

 

def frange(start, stop, step):
      i = start
      while i < stop:
          yield i
          i += step

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

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
    denom = 1 + zReal**2 + zImag**2
    x = 2*zReal / denom
    y = -2*zImag / denom
    z = (1 - zReal**2 - zImag**2) / denom
    xyz = np.array( [x,y,z] )
    return xyz

def frac(n):
    num = 1
    numVec = np.zeros(n)
    #numVec[0] = 1
    while num < (n-1):
        #print(num)
        numVec[num-1] = 1/num
        num += 1
    return numVec

def sq2series(n):
    num = 1
    while num < n:
        sq2s = 1/(num**2)
        yield sq2s
        num += 1

def sumSq2(n):
    num = 1
    numVec = np.zeros(n-1)
    while num < n:
        #if num != 12345678910111213:
        if num != 3:
           numVec[num-1] = 1/(num*np.sqrt(2))**2 + 1/(num*np.sqrt(2))**2
        else:
            print("Number is Three.")
            #numVec[num-1] = 1/(num*np.sqrt(2))**2 + 1/(num*np.sqrt(2))**2
        num += 1
    return numVec

# Vectorize
    #zetavec = np.vectorize(zeta)
    #etavec = np.vectorize(eta)
    #xyzunitvec = np.vectorize(xyzunit)

#fracVal = np.zeros(100)

fracVal = np.sqrt(2)*frac(100) 

          #zetaVal

sqOutput = sq2series(10000)

sqValue = sum(sqOutput)
print("Ordinary squared series:   {}".format(sqValue))

xvec = np.array([ 1/2 ]* len(fracVal))
yvec = np.array([ np.sqrt(3)/2 ]* len(fracVal))

zvec = xvec

xyvec = np.array( [ complex(0,0) ] * len(xvec) )
xyvec.real = xvec; xyvec.imag = yvec
zetavec = np.array( [ complex(0,0) ] * len(xvec) )
zetavec.real = xvec; zetavec.imag = yvec;

zetavec = fracVal*zetavec

zetapolar = polarvec(zetavec.real,zetavec.imag)

#zvec



zCalc = xyvec * zetavec.conj()
zCalx = 1 - zCalc

angleVec = np.array([1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,
12/12,1/8,3/8,5/8,7/8])

# Decimal values for cos of popular Angles
angleVec = np.sort(angleVec,axis=0)
    
seriesMat = np.zeros(len(xvec),dtype=[('xvec','f8'),('yvec','f8'),
('zvec','f8'),('ZetaR','f8'),('ZetaI','f8'),('EtaR','f8'),
('EtaI','f8')])

seriesMat['xvec'] = xvec
seriesMat['yvec'] = yvec
seriesMat['zvec'] = zCalx.real

plotRows = len(xvec)
angleLine = np.zeros([plotRows,len(angleVec)])

for ii in range(0,len(angleVec)):
    angleLine[:,ii] = np.array( [angleVec[ii]] * plotRows)
    


plt.figure(1)
plt.plot(seriesMat['zvec'],'bo',markersize=3.5)

for ii in range(0,len(angleLine[0])):
    plt.plot(angleLine[:,ii],'r',linewidth=1.5)

# Do the Square to Sum Function
    
sq2sumA = sum(sumSq2(10000))
sqThree = 1/(3*np.sqrt(3))**2 + (np.sqrt(2)/(3*np.sqrt(3)))**2
print("   ")
print("The sum of Square Two's:   {}".format(sq2sumA))
print("   ")
print("Square Three:   {}".format(sqThree))
print("   ")
print("Pi^2 over 6:    {}".format(np.pi**2/6))

#Zeta=zetavec(riemMat['xvec'],riemMat['yvec'],riemMat['zvec'])
#Eta=etavec(riemMat['xvec'],riemMat['yvec'],riemMat['zvec'])
#riemMat['ZetaR'] = Zeta.real
#riemMat['ZetaI'] = Zeta.imag
#riemMat['EtaR'] = Eta.real
#riemMat['EtaI'] = Eta.imag




