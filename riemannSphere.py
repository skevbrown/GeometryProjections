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

# Vectorize
zetavec = np.vectorize(zeta)
etavec = np.vectorize(eta)
xyzunitvec = np.vectorize(xyzunit)







################ Latitude ########################
azimindex = 30; polindex = int( (azimindex)/2 );

radiusConst = 1.0

   # Set lower/ x and y, z limits
xl = -1.0; xh = 1.0; yl = xl; yh = xh;
zl = xl; zh = xh;


azimincr = 2*np.pi/azimindex; 
polincr = np.pi/polindex;
azimarray = np.arange(0,         2*np.pi,   azimincr);
polarray =  np.arange(-np.pi/2,  np.pi/2,    polincr);

azimuth = [ 0 ] * (azimindex*polindex);
polar =   [ 0 ] * (azimindex*polindex);
rad = [ radiusConst ] * (azimindex*polindex);

ind = 0; # Fill arrays for azimuth, polar, etc with the right plot order
for i in range(0, len(polarray) ):
    for ii in range(0, len(azimarray) ):
        azimuth[ind] = azimarray[ii];
        polar[ind]   = polarray[i];  
        ind = ind+1;

matrows = azimindex; matcols = polindex*3;
sphere = np.zeros( [matrows, matcols ] );

i = 0; # Fill sphere matrix in col groups of 3 to plot in col groups of 3
for ii in range(2, matcols ,3): 
    rowlower = 0+i*azimindex; rowhigher = azimindex+i*azimindex - 1
    #print(rowlower); print(rowhigher); 
    sphere[0:matrows,ii-2] = azimuth[rowlower:rowhigher+1];
    #print(azimuth[rowlower:rowhigher+1]); print(sphere[0:azimindex,ii-2] ); print("");
    sphere[0:matrows,ii-1] = polar[rowlower:rowhigher+1];
    sphere[0:matrows,ii  ] = rad[rowlower:  rowhigher+1];
    i = i+1;


xyz = np.zeros( [matrows, matcols ] );

for ii in range(2, polindex*3, 3): # Now convert sphere to cart in col gropus of three
    spheresub = sph2cartvec( sphere[:,ii-2], sphere[:,ii-1], sphere[:,ii] );
    #print(spheresub );
    xyz[:,ii-2] = spheresub[0];
    xyz[:,ii-1] = spheresub[1];
    xyz[:,ii]   = spheresub[2];




################### Longitude #####################

#azimincr = 2*np.pi/azimindex; 
#polincr = np.pi/polindex;
#azimarray = np.arange(0,         2*np.pi,   azimincr);
#polarray =  np.arange(-np.pi/2,  np.pi/2,    polincr);

azimuthLon = [ 0 ] * (azimindex*polindex);
polarLon =   [ 0 ] * (azimindex*polindex);
radLon = [ 0.5 ] * (azimindex*polindex);

ind = 0; # Fill arrays for azimuth, polar, etc with the right plot order
for i in range(0, len(azimarray) ):
    for ii in range(0, len(polarray) ):
        azimuthLon[ind] = azimarray[i];
        polarLon[ind]   = polarray[ii];  
        ind = ind+1;


matrows = polindex; matcols = azimindex*3;
sphereLon = np.zeros( [matrows, matcols ] );

i = 0; # Fill sphere in col groups of 3
for ii in range(2, matcols ,3): 
    rowlower = 0+i*matrows; rowhigher = matrows+i*matrows - 1
         #print(rowlower); print(rowhigher); 
    sphereLon[0:matrows,ii-2] = azimuthLon[rowlower:rowhigher+1];
    #print(polar[rowlower:rowhigher+1]); print(sphere[0:azimindex,ii-2] ); print("");
    sphereLon[0:matrows,ii-1] = polarLon[rowlower:rowhigher+1];
    sphereLon[0:matrows,ii  ] = rad[rowlower:  rowhigher+1];
    i = i+1;


xyzLon = np.zeros( [matrows, matcols ] );

for ii in range(2, matcols, 3): # Now convert sphere to cart in col gropus of three
    spheresub = sph2cartvec( sphereLon[:,ii-2], sphereLon[:,ii-1], sphereLon[:,ii] );
       #print(spheresub );
    xyzLon[:,ii-2] = spheresub[0];
    xyzLon[:,ii-1] = spheresub[1];
    xyzLon[:,ii]   = spheresub[2];


############## Random Point Groupings ##############

groupsize = 20; percentRng = 0.1

numRandLoc = 20;
randomLocate = np.zeros( [numRandLoc,2] )
for i in range(0,numRandLoc): 
    randomLocate[i,0] = uniform(0, 2*np.pi);
    randomLocate[i,1] = uniform(-np.pi/2, np.pi/2);

# Use random locations in cartesian to get better pos neg balance
randomLocxyz = np.zeros( [len(randomLocate), 3] );
sphereout = sph2cartvec( abs(randomLocate[:,0]), abs(randomLocate[:,1]), [ radiusConst ] * len(randomLocate) ) 

posneg1 = [ 0 ] *len(sphereout[0] );
posneg2 = [ 0 ] *len(sphereout[1]);
posneg3 = [ 0 ] *len(sphereout[2]);
for i in range(0, len(sphereout[0]) ):
    u1 = randint(0,1); 
    if u1 == 0: 
          u1 = -1; 
    posneg1[i] = u1;
    u2 = randint(0,1); 
    if u2 == 0: 
       u2 = -1; 
    posneg2[i] = u2;
    u3 = randint(0,1); 
    if u3 == 0: 
       u3 = -1; 
    posneg3[i] = u3;


sphereoutZero = sphereout[0] *posneg1;
sphereoutOne =  sphereout[1] *posneg2;
sphereoutTwo =  sphereout[2] *posneg3;

xyzout = cart2sphvec(sphereoutZero, sphereoutOne, sphereoutTwo );

randomLocate[:,0] = xyzout[0];
randomLocate[:,1] = xyzout[1];
#randomLocate[:,2] = xyzout[2];



matrows = groupsize*numRandLoc; matcols = 3;
sphereRand = np.zeros([ matrows, matcols ] );
ind = 0;
for i in range(0,numRandLoc): 
    locAzim = randomLocate[i,0];
    locPolar = randomLocate[i,0];
    
    for ii in range(0,groupsize):
        locAzimPlus = locAzim + uniform( 0, percentRng*2*np.pi);
        locPolarPlus = locPolar + uniform( -percentRng*np.pi/2, percentRng*np.pi/2);
   

 
        sphereRand[ind,0] = locAzimPlus;
        sphereRand[ind,1] = locPolarPlus;
        sphereRand[ind,2] = radiusConst;
        ind = ind + 1;

xyzRandArray = sph2cartvec(sphereRand[:,0], sphereRand[:,1], sphereRand[:,2] );

xyzRand = np.zeros([ matrows, matcols ] );

xyzRand[:,0] = xyzRandArray[0] ;
xyzRand[:,1] = xyzRandArray[1] ;
xyzRand[:,2] = xyzRandArray[2] ;

azimtest = np.arange(0,2*np.pi, 2*np.pi/50);
poltest = [ 0.524 ] * azimtest.size;
radtest = [ 1.0 ] * azimtest.size;

xyztest0, xyztest1, xyztest2 = sph2cartvec(azimtest, poltest, radtest)
xyztest = np.zeros( [xyztest0.size,3] )
xyztest[:,0] = xyztest0; xyztest[:,1] = xyztest1; xyztest[:,2] = xyztest2;

xyzzeros = np.zeros( [xyzRand.size + xyztest.size, 3] )

xyzzeros = np.concatenate( (xyzRand,xyztest), axis=0 )

del xyzRand; xyzRand = xyzzeros;


  # Translate sphere and points 1/2 unit on z-axis
  # Can use this for translation by adding to xyzRand
ztrans = np.zeros( [len(xyz),3] ); ztrans[:,2] = 0.5;
ztrans1 = ztrans;

for i in range(0, int(len(xyz[0])/3)-1 ):
    ztrans = np.concatenate( (ztrans,ztrans1), axis=1 );
# xyz = xyz + ztrans;


ztrans = np.zeros( [len(xyzLon),3] ); ztrans[:,2] = 0.5;
ztrans1 = ztrans;

for i in range(0, int(len(xyzLon[0])/3)-1 ):
    ztrans = np.concatenate( (ztrans,ztrans1), axis=1 );
# xyzLon = xyzLon + ztrans;


ztrans = np.zeros( [len(xyzRand), len(xyzRand[0]) ] );
ztrans[:,2] = 0.5;

# xyzRand = xyzRand + ztrans;



xgrid = np.arange(-1.0,1.4,0.4); ygrid = xgrid;




############# Riemann Sphere Plot ##################

#fig2 = plt.figure(2)          # A static 3d plot
#ax2 = fig2.gca(projection='polar')
#ax2 = set_rmax(20)


              # Could be 1/6 2 sqrt 3 or 1/3
point1 = np.array([0,0,1/3])
normal1 = np.array([0,0,1.0])

xx1, yy1 = np.meshgrid(range(2),range(2))

d1 = -np.sum(point1*normal1)
z1 = (-normal1[0]*xx1 - normal1[1]*yy1 - d1)*1/normal1[2]




fig1 = plt.figure(1)          # A static 3d plot
ax = fig1.gca(projection='3d')
for ii in range(2, polindex*3, 3):
    ax.plot(xyz[:,ii-2], xyz[:,ii-1], xyz[:,ii], color=(0.0,0.0,0.0,0.5), 
    linewidth=0.3)

for ii in range(2, azimindex*3, 3):
    ax.plot(xyzLon[:,ii-2], xyzLon[:,ii-1], xyzLon[:,ii], 
    c=(0.0,0.0,0.0,0.5), linewidth=0.5)

ax.scatter(xyzRand[:,0], xyzRand[:,1], xyzRand[:,2], s=1.2,  
c=(0.2,0.2,0.8,0.8) )

for ii in range(0, len(xgrid) ):
    ax.plot( [ xgrid[ii] ] *len(ygrid), ygrid, [ 0 ] * len(ygrid), 
    c=(0.8,0.0,0.0,0.5), linewidth=1.0)

for ii in range(0, len(ygrid) ):
    ax.plot( xgrid, [ ygrid[ii] ] *len(xgrid), [ 0 ] * len(ygrid), 
    c=(0.8,0.0,0.0,0.5), linewidth=1.0)


ax.set_xlim(xl, xh); ax.set_ylim(yl, yh);
ax.set_zlim(zl, zh);

# Calc's to rotate the Cube
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
magLS = mag3D(lineStart)
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
    rowOut = np.dot(rotation_matrix(axis,theta), list(cubeRot1[ii]))
    cubeRot2[ii] = rowOut
    
check_orthog(0,1,3,cubeRot2)
lineStart = np.array(list(cubeRot2[11])) - np.array(list(cubeRot2[10]))
magLS = mag3D(lineStart)
print("Final Diag length {}".format(magLS))

for row in range(0,len(cubeRot2)):
    magLS = mag3D(np.array(list(cubeRot2[row])))
    #sphCoor = cart2sph(np.array(list(cubeRot2[row])))
    print("Mag of point {} {}".format(row,magLS))
    #print("Sph {} {} {}".format(sphCoor[0],sphCoor[1]),sphCoor[2])

# Plane intersection points
vecCenter =  np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
vecCorner1 = np.array([0,2/np.sqrt(3),2/np.sqrt(3)])
vecCornxyz1 = vecCorner1 - vecCenter

vecCorner2 = np.array([2/np.sqrt(3),0,2/np.sqrt(3)])
vecCornxyz2 = vecCorner2 - vecCenter

vecCorner3 = np.array([2/np.sqrt(3),2/np.sqrt(3),0])
vecCornxyz3 = vecCorner3 - vecCenter
vecCornxyz3 = vecCornxyz3 * [-1,-1,-1]


fig2 = plt.figure(2)          # A static 3d plot
ax = fig2.gca(projection='3d')
for ii in range(2, polindex*3, 3):
    ax.plot(xyz[:,ii-2], xyz[:,ii-1], xyz[:,ii], color=(0.0,0.0,0.0,0.5), 
    linewidth=0.3)

for ii in range(2, azimindex*3, 3):
    ax.plot(xyzLon[:,ii-2], xyzLon[:,ii-1], xyzLon[:,ii], 
    c=(0.0,0.0,0.0,0.5), linewidth=0.5)

for ii in range(0, len(xgrid) ):
    ax.plot( [ xgrid[ii] ] *len(ygrid), ygrid, [ 0 ] * len(ygrid), 
    c=(0.8,0.0,0.0,0.5), linewidth=1.0)

for ii in range(0, len(ygrid) ):
    ax.plot( xgrid, [ ygrid[ii] ] *len(xgrid), [ 0 ] * len(ygrid), 
    c=(0.8,0.0,0.0,0.5), linewidth=1.0)

# Four surface to make One plane: Upper plane
ax.plot_surface(xx1,yy1,z1,color=(0.2,0.1,0.9,0.3))
ax.plot_surface(-xx1,yy1,z1,color=(0.2,0.1,0.9,0.3))
ax.plot_surface(xx1,-yy1,z1,color=(0.2,0.1,0.9,0.3))
ax.plot_surface(-xx1,-yy1,z1,color=(0.2,0.1,0.9,0.3))
# Lower plane
ax.plot_surface(xx1,yy1,-z1,color=(0.9,0.1,0.2,0.3))
ax.plot_surface(-xx1,yy1,-z1,color=(0.9,0.1,0.2,0.3))
ax.plot_surface(xx1,-yy1,-z1,color=(0.9,0.1,0.2,0.3))
ax.plot_surface(-xx1,-yy1,-z1,color=(0.9,0.1,0.2,0.3))

# Scatter points
#ax.scatter(vecCornxyz1[0],vecCornxyz1[1],vecCornxyz1[2],s=24.2,
#c=(0.9,0.1,0.1,1.0))
#ax.scatter(vecCornxyz2[0],vecCornxyz2[1],vecCornxyz2[2],s=24.2,
#c=(0.9,0.1,0.1,1.0))
#ax.scatter(vecCornxyz3[0],vecCornxyz3[1],vecCornxyz3[2],s=24.2,
#c=(0.9,0.1,0.1,1.0))

# Plot the rotated cube
cubePlot = np.copy(cubeRot2)

ax.scatter(cubePlot['xPnts'][0:9],cubePlot['yPnts'][0:9],
cubePlot['zPnts'][0:9],color='blue',s=54.2)
#ax.plot(cubePlot['xPnts'][0:5],cubePlot['yPnts'][0:5],
#cubePlot['zPnts'][0:5],label='Cube',color='blue',linewidth=0.7)
#ax.plot(cubePlot['xPnts'][5:10],cubePlot['yPnts'][5:10],
#cubePlot['zPnts'][5:10],label='Cube',color='blue',linewidth=0.7)
ax.plot(cubePlot['xPnts'][10:12],cubePlot['yPnts'][10:12],
cubePlot['zPnts'][10:12],color='red',linewidth=1.5)

ax.set_xlim(xl, xh); ax.set_ylim(yl, yh);
ax.set_zlim(zl, zh);


#fig3 = plt.figure(3)
#ax2 = fig3.gca(projection='polar' )

xyzRad = np.sqrt(xyzRand[:,0]**2 + xyzRand[:,1]**2 ); 
xyzAng = np.arcsin(xyzRand[:,1]/xyzRad );

#ax1 = plt.subplot(111, projection='polar')
#ax2.scatter(xyzAng,  xyzRad,  lw=0.5, c='red' )



############ ZETA VALUES FROM X,Y,Z ###################
############ AND POLAR PLOT ###########################


zetaArray = zetavec(xyzRand[:,0], xyzRand[:,1], xyzRand[:,2] )
zetaMag = abs(zetaArray); zetaPhase = np.arctan(zetaArray.imag/zetaArray.real);

for i in range(0, zetaMag.size ):
    if zetaMag[i] > 10:
         zetaMag[i] = 8; 


#fig3 = plt.figure(3)
#ax2 = fig3.gca(projection='polar' )

xyzRad = np.sqrt(xyzRand[:,0]**2 + xyzRand[:,1]**2 ); 
xyzAng = np.arctan(xyzRand[:,1]/xyzRand[:,0] );

#ax1 = plt.subplot(111, projection='polar')
#ax2.scatter(zetaPhase, zetaMag, lw=0.5, c='blue' )
#ax2.scatter(xyzAng,  xyzRad,    lw=0.5, c='red' )




#fig3 = plt.figure(3)

#plt.plot(np.real(zetaArray), 'b' )
#plt.plot(np.imag(zetaArray), 'r' )

zvec = np.arange(-0.75,0.75,1.5/200)
yvec = np.array( [ 0 ] * len(zvec) )
xvec = np.sqrt(1 - np.abs(zvec**2) + np.abs(yvec**2))

riemMat = np.zeros(len(xvec),dtype=[('xvec','f8'),('yvec','f8'),
('zvec','f8'),('ZetaR','f8'),('ZetaI','f8'),('EtaR','f8'),
('EtaI','f8')])

riemMat['xvec'] = xvec
riemMat['yvec'] = yvec
riemMat['zvec'] = zvec

Zeta=zetavec(riemMat['xvec'],riemMat['yvec'],riemMat['zvec'])
Eta=etavec(riemMat['xvec'],riemMat['yvec'],riemMat['zvec'])
riemMat['ZetaR'] = Zeta.real
riemMat['ZetaI'] = Zeta.imag
riemMat['EtaR'] = Eta.real
riemMat['EtaI'] = Eta.imag

fig4 = plt.figure(4)

plt.plot(riemMat['xvec'],'black',linewidth=1.1)
plt.plot(riemMat['zvec'],'b',linewidth=0.7)
plt.plot(riemMat['ZetaR'],'r',linewidth=0.7)
plt.plot(riemMat['EtaR'],'g',linewidth=0.7)
#plt.ylim([-1.0,5.0])



zetatest = np.array( [ (1.2+0.3j), (1.0+0j), (0.3+0.5j), (1.0+3.0j), (10+30j), (4000+10j), (3000+3000j) ] )

zetatestR = zetatest.real; zetatestI = zetatest.imag;
xvectest = [ 0 ]*len(zetatestR); yvectest = [ 0 ]*len(zetatestR); 
zvectest = [ 0 ]*len(zetatestR);

for i in range(0, len(zetatestR) ):
    xvectest[i], yvectest[i], zvectest[i] = xyzunit(zetatestR[i], zetatestI[i] )

 


plt.show()



#fig, ax = plt.subplots()     # Setup for animated plot
#line, = ax.plot([], [], lw=1 )

#xdata, ydata = [], []

#ax.grid()



#def run(data): # This is the data update function (sets new data)
        # update the data
#    t, y = data
#    xdata.append(t)
#    ydata.append(y)
#    xmin, xmax = ax.get_xlim()

        #if t >= xmax:                  # This will readjust axis as time vector points increase
        #    ax.set_xlim(xmin, 2*xmax)
        #    ax.figure.canvas.draw()
#    line.set_data(xdata, ydata)

#    return line,

#ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=20,repeat=False, init_func=init)


