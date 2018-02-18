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
plt3d = plt.figure(1).gca(projection='3d')
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



bv = 1/np.sqrt(3)# Base value of matrix, Gives Diag of 2.0
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

# A: Rotate around X by +pi/4, Rotate around Y by -(1/5)*pi
# B: Rotate around Y by -pi/4, Rotate around X by 1/5*pi
# COEFF'S ARE SAME IN ROT MATRICES IN TERMS OF POSITIONS, BUT
# THE SIGN OF SOME COEFF'S CHANGES

v = [3, 5, 0]
axis = [0,1,0] # Rotate around Y, by pi/4
theta = -np.pi/4 

print(rotation_matrix(axis,theta))

for ii in range(0,len(cubePoints)):
    rowOut = np.dot(rotation_matrix(axis,theta), list(cubePoints[ii]))
    cubeRot1[ii] = rowOut
    print(rowOut ) 
    
check_orthog(0,1,3,cubeRot1)

    
v = [3, 5, 0]
axis = [1,0,0] # Rotate around X by pi/5
#theta = 1/np.sqrt(3) # Eigth/Fortieths ???
theta = 0.62168  # -0.62168

print(rotation_matrix(axis,theta))

for ii in range(0,len(cubeRot1)):
            #print(cubePoints[ii])
    rowOut = np.dot(rotation_matrix(axis,theta), list(cubeRot1[ii]))
    cubeRot2[ii] = rowOut 
    
check_orthog(0,1,3,cubeRot2)
lineStart = np.array(list(cubeRot2[11])) - np.array(list(cubeRot2[10]))
magLS = np.sqrt(lineStart[0]**2+lineStart[1]**2+lineStart[2]**2)
print("Final Diag length {}".format(magLS))

cubePlot = np.copy(cubeRot2)

# Non Rotated Cube
plt3d = plt.figure(2).gca(projection='3d')

plt3d.scatter(cubePoints['xPnts'][0:9],cubePoints['yPnts'][0:9],
cubePoints['zPnts'][0:9],color='red')
plt3d.plot(cubePoints['xPnts'][0:5],cubePoints['yPnts'][0:5],
cubePoints['zPnts'][0:5],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][5:10],cubePoints['yPnts'][5:10],
cubePoints['zPnts'][5:10],label='Cube',color='blue',linewidth=0.7)
plt3d.plot(cubePoints['xPnts'][10:12],cubePoints['yPnts'][10:12],
cubePoints['zPnts'][10:12],color='red',linewidth=1.5)

# Rotated Cube
plt3dRot = plt.figure(3).gca(projection='3d')

plt3dRot.scatter(cubePlot['xPnts'][0:9],cubePlot['yPnts'][0:9],
cubePlot['zPnts'][0:9],color='red')
plt3dRot.plot(cubePlot['xPnts'][0:5],cubePlot['yPnts'][0:5],
cubePlot['zPnts'][0:5],label='Cube',color='blue',linewidth=0.7)
plt3dRot.plot(cubePlot['xPnts'][5:10],cubePlot['yPnts'][5:10],
cubePlot['zPnts'][5:10],label='Cube',color='blue',linewidth=0.7)
plt3dRot.plot(cubePlot['xPnts'][10:12],cubePlot['yPnts'][10:12],
cubePlot['zPnts'][10:12],color='red',linewidth=1.5)


# Rotated cube Drawn
# Specify rotated cube in spherical, put in Structure, Convert and put in XYZ 
# structured
trAngle = 1 - np.arcsin(1/np.sqrt(3))
cubedrawMat = np.array([[0,np.pi/2,1.0],[0,-np.pi/2,1.0], # Top bottom
[-(1/6)*np.pi,-trAngle,1.0],[(1/6)*np.pi,trAngle,1.0],[np.pi/2,-trAngle,1.0],
[(5/6)*np.pi,trAngle,1.0],[-(5/6)*np.pi,-trAngle,1.0],[-(np.pi/2),trAngle,1.0]])  # Upper triang

# Setup a circle at Z=0
circleAz = np.arange(-np.pi,np.pi,2*np.pi/100)
circlePolar = np.array( [ 0.0] * len(circleAz))
circleRad = np.array( [1.0]*len(circleAz) )

circleStruct = np.zeros(len(circleAz),dtype=[('xPnts','f8'),('yPnts','f8'),
('zPnts','f8')])

for row in range(0,len(circleStruct)):
    xpnt, ypnt, zpnt = sph2cart(circleAz[row],circlePolar[row],circleRad[row])
    circleStruct['xPnts'][row] = xpnt
    circleStruct['yPnts'][row] = ypnt
    circleStruct['zPnts'][row] = zpnt


cubeDrawnSp = np.zeros(len(cubedrawMat),dtype=[('azim','f8'),('polar','f8'),('radius','f8')])
cubeDrawn = np.zeros(len(cubedrawMat),dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])

cubeDrawnSp['azim'] = cubedrawMat[:,0]
cubeDrawnSp['polar'] = cubedrawMat[:,1]
cubeDrawnSp['radius'] = cubedrawMat[:,2]

for row in range(0,len(cubeDrawnSp)):
    azim = cubeDrawnSp['azim'][row]
    polar = cubeDrawnSp['polar'][row]
    radius = cubeDrawnSp['radius'][row]
    xpnt, ypnt, zpnt = sph2cart(azim,polar,radius)
    cubeDrawn['xPnts'][row] = xpnt
    cubeDrawn['yPnts'][row] = ypnt
    cubeDrawn['zPnts'][row] = zpnt


# Rotated each point in Cube to get rotated cube with 
# Diag as center axis with length of 2 (-1.0 to 1.0)

# Setup a plane for plot
point1 = np.array([0,0,0])
normal1 = np.array([0,0,1])

xx, yy = np.meshgrid(range(2),range(2))

# Rotate all points with Normal vector
d1 = -np.sum(point1*normal1) # Dot product
z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1/normal1[2]

# Individual lines to add to plot
mainAxis = np.array( [np.array(list(cubeDrawn[0])),
np.array(list(cubeDrawn[1]))])

line1_4 = np.array( [np.array(list(cubeDrawn[1])),
np.array(list(cubeDrawn[4]))])
line0_7 = np.array( [np.array(list(cubeDrawn[0])),
np.array(list(cubeDrawn[7]))])
line0_5 = np.array( [np.array(list(cubeDrawn[0])),
np.array(list(cubeDrawn[5]))])
    
lines = np.concatenate((line0_7,line1_4),axis=0)    
#lines = np.concatenate((lines,line0_5),axis=0)
    
# Plot Drawn(rotated) Cube
plt3dRaw = plt.figure(4).gca(projection='3d')

plt3dRaw.scatter(cubeDrawn['xPnts'],cubeDrawn['yPnts'],
cubeDrawn['zPnts'],color='red',s=54.2)
plt3dRaw.plot(mainAxis[:,0],mainAxis[:,1],mainAxis[:,2],color='red',
linewidth=1.5)
#plt3dRaw.plot(lines[:,0],lines[:,1],lines[:,2],color='blue',
#linewidth=0.7)
#plt3dRaw.plot(line0_5[:,0],line0_5[:,1],line0_5[:,2],color='blue',
#linewidth=0.7)
plt3dRaw.plot(circleStruct['xPnts'],circleStruct['yPnts'],
circleStruct['zPnts'],color='blue',linewidth=0.7)

for row in range(0,len(cubeDrawn)-1):
    lineVec1 = np.array(list(cubeDrawn[row]))
    lineVec2 = np.array(list(cubeDrawn[row+1]))
    lineMat =  np.array([lineVec1,lineVec2])
    #print(lineMat)
    plt3dRaw.plot(lineMat[:,0],lineMat[:,1],lineMat[:,2],
    color='blue',linewidth=0.7)

# Four surface to make One plane: Upper plane
plt3dRaw.plot_surface(xx,yy,z1,color=(0.2,0.1,0.9,0.3))
plt3dRaw.plot_surface(-xx,yy,z1,color=(0.2,0.1,0.9,0.3))
plt3dRaw.plot_surface(xx,-yy,z1,color=(0.2,0.1,0.9,0.3))
plt3dRaw.plot_surface(-xx,-yy,z1,color=(0.2,0.1,0.9,0.3))
#plt3d.set_xlim(-2,2); plt3d.set_ylim(-2,2)
#plt3d.set_zlim(0,1.2)

# Rotated Cube
plt3dTriang = plt.figure(5).gca(projection='3d')

# A circle
# sph2cartvec
azimCircle = np.arange(0,2*np.pi,2*np.pi/300)
polCircle = np.array([ 0.0 ]*len(azimCircle))
radCircle = np.array([ 1.0 ]*len(azimCircle))

circleMat = np.zeros(len(azimCircle),dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])
xConv, yConv, zConv = sph2cartvec(azimCircle,polCircle,radCircle)
circleMat['xPnts'] = xConv; circleMat['yPnts'] = yConv;
circleMat['zPnts'] = zConv

#lineMat2 = np.zeros([4])
lineMat2 = np.zeros(12,dtype=[('xPnts','f8'),('yPnts','f8'),('zPnts','f8')])

# Four surface to make One plane: Upper plane
plt3dTriang.plot_surface(xx,yy,z1,  color=(0.2,0.1,0.9,0.2))
plt3dTriang.plot_surface(-xx,yy,z1, color=(0.2,0.1,0.9,0.2))
plt3dTriang.plot_surface(xx,-yy,z1, color=(0.2,0.1,0.9,0.2))
plt3dTriang.plot_surface(-xx,-yy,z1,color=(0.2,0.1,0.9,0.2))

twoSq2 = 2*np.sqrt(2) 

 
lineMat2[0] = np.array([0,0,0])
lineMat2[1] = np.array([np.cos(np.pi/3)*twoSq2/3,np.sin(np.pi/3)*twoSq2/3,0])
plt3dTriang.plot(lineMat2['xPnts'][0:2],lineMat2['yPnts'][0:2],
lineMat2['zPnts'][0:2],color='blue',linewidth=1.5)

lineMat2[2] = np.array([0,0,0])
lineMat2[3] = np.array([np.cos(np.pi/3)*twoSq2/3,0,0])
plt3dTriang.plot(lineMat2['xPnts'][2:4],lineMat2['yPnts'][2:4],
lineMat2['zPnts'][2:4],color='red',linewidth=1.5)

lineMat2[4] = np.array([0,0,0])
lineMat2[5] = np.array([np.cos(np.pi/3)*twoSq2/3,np.sin(np.pi/3)*twoSq2/3,1/3])
plt3dTriang.plot(lineMat2['xPnts'][4:6],lineMat2['yPnts'][4:6],
lineMat2['zPnts'][4:6],color='blue',linewidth=1.5)

lineMat2[6] = np.array([0,0,-1/2])
lineMat2[7] = np.array([0,0,1/2])
plt3dTriang.plot(lineMat2['xPnts'][6:8],lineMat2['yPnts'][6:8],
lineMat2['zPnts'][6:8],color='black',linewidth=1.5)

lineMat2[8] = np.array([np.cos(np.pi/3)*twoSq2/3,np.sin(np.pi/3)*twoSq2/3,0])
lineMat2[9] = np.array([np.cos(np.pi/3)*twoSq2/3,np.sin(np.pi/3)*twoSq2/3,1/3])
plt3dTriang.plot(lineMat2['xPnts'][8:10],lineMat2['yPnts'][8:10],
lineMat2['zPnts'][8:10],color='blue',linewidth=1.5)

lineMat2[10] = np.array([np.cos(np.pi/3)*twoSq2/3,0,0])
lineMat2[11] = np.array([np.cos(np.pi/3)*twoSq2/3,np.sin(np.pi/3)*twoSq2/3,0])
plt3dTriang.plot(lineMat2['xPnts'][10:12],lineMat2['yPnts'][10:12],
lineMat2['zPnts'][10:12],color='red',linewidth=1.5)


plt3dTriang.scatter(circleMat['xPnts'],circleMat['yPnts'],circleMat['zPnts'],
color='blue',s=0.5)


plt.show()




