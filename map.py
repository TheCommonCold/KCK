from __future__ import division
import matplotlib
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import math
import colorsys

def openfile():
    with open('big.dem.txt') as f:
        w, h, scale = [int(x) for x in next(f).split()]
        data=[[float(x) for x in line.split()] for line in f]
    return data
def interpolator(hsv1, hsv2, t):
    return (hsv1[0] + (hsv2[0] - hsv1[0]) * t), (hsv1[1] + (hsv2[1] - hsv1[1]) * t), (hsv1[2] + (hsv2[2] - hsv1[2]) * t)

def normalize(x):
    x = x/(math.sqrt((x[0] **2) + (x[1] **2) + (x[2] **2)))
    return x

def calcangle(a):
    b=(0, 0.447214, 0.894427)
    return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/(math.sqrt(a[0]**2+a[1]**2+a[2]**2)*math.sqrt(b[0]**2+b[1]**2+b[2]**2))

def colorizer(data,angles):
    temp=[]
    for array in data:
        temp.append(min(array))
        temp.append(max(array))
    minimum=min(temp)
    maximum=max(temp)
    for i in range(len(data)):
        for j in range(len(data[0])):
            #if(angles[i][j])>0:
            #    data[i][j]=gradient((data[i][j] - minimum) / (maximum - minimum), 1 *angles[i][j], 1)
            #else:
            #    data[i][j] = gradient((data[i][j] - minimum) / (maximum - minimum), 1, 1/abs(angles[i][j]))

            if (angles[i][j]) > 0:
                data[i][j] = gradient((data[i][j] - minimum) / (maximum - minimum), 1*(angles[i][j])**2, 1 / (1/(angles[i][j])**4)+0.1)
            else:
                data[i][j] = gradient((data[i][j] - minimum) / (maximum - minimum), 1, 1 )
    return data

def crossvectors(a,b,c,d):
    return normalize(np.cross(a,b)+np.cross(b,c)+np.cross(c,d)+np.cross(d,a))

def normalcalculator(data):
    normals=[]
    angles=[]
    angles.append(np.ones((len(data[0]))))
    for i in range(1,len(data)-1):
        tempnormals=[]
        tempangles=[1]
        for j in range(1,len(data[0])-1):
            tempnormals.append(crossvectors((75.31,0,data[i][j+1]),(0,75.31,data[i-1][j]),(-75.31,0,data[i][j-1]),(0,-75.31,data[i+1][j])))
            tempangles.append(calcangle(tempnormals[j-1]))
        tempangles.append(1);
        angles.append(tempangles)
    angles.append(np.ones((len(data[0]))))
    return angles


def gradient(x,saturation,level):
    colour1=(10/30,saturation,level)
    colour2=(6/36,saturation,level)
    colour3=(0,saturation,level)
    if (x<1/2):
        return interpolator(hsv_to_rgb(colour1), hsv_to_rgb(colour2), (x*2))
    else:
        return interpolator(hsv_to_rgb(colour2), hsv_to_rgb(colour3), (x - 1/2) * 2)



if __name__ == '__main__':
    data=openfile()
    #img=plt.imshow(data,cmap='nipy_spectral',clim=(-80,170), interpolation='nearest')
    #print(img)
    #plt.colorbar()

    angles=normalcalculator(data)
    data = colorizer(data,angles)
    plt.imshow(data)
    plt.show()