from __future__ import division
import matplotlib
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import colorsys

def openfile():
    with open('big.dem.txt') as f:
        w, h, scale = [int(x) for x in next(f).split()]
        data=[[float(x) for x in line.split()] for line in f]
    return data
def interpolator(hsv1, hsv2, t):
    return (hsv1[0] + (hsv2[0] - hsv1[0]) * t), (hsv1[1] + (hsv2[1] - hsv1[1]) * t), (hsv1[2] + (hsv2[2] - hsv1[2]) * t)

def normalize(x):
    return (x - min(x)) / (max(x)-min(x))

def calcangle(a):
    b=(-0.157991, -0.789953, -0.592464)
    a[]
def normalizer(data):
    temp=[]
    for array in data:
        temp.append(min(array))
        temp.append(max(array))
    minimum=min(temp)
    maximum=max(temp)
    data=[[gradient((x-minimum)/(maximum-minimum))for x in array] for array in data]
    return data

def crossvectors(a,b,c,d):
    return normalize(np.cross(a,b)+np.cross(b,c)+np.cross(c,d)+np.cross(d,a))

def normalcalculator(data):
    normals=[]
    for i in range(1,len(data)-1):
        temparray=[]
        for j in range(1,len(data[0])-1):
            temparray.append(crossvectors((75.31,0,data[i][j+1]-data[i][j]),(0,75.31,data[i-1][j]-data[i][j]),(-75.31,0,data[i][j-1]-data[i][j]),(0,-75.31,data[i+1][j]-data[i][j])))
        #print(temparray)
        normals.append(temparray);
def gradient(x):
    colour1=(10/30,1,0.8)
    colour2=(6/36,1,1)
    colour3=(0,1,1)
    if (x<1/2):
        return interpolator(hsv_to_rgb(colour1), hsv_to_rgb(colour2), (x*2))
    else:
        return interpolator(hsv_to_rgb(colour2), hsv_to_rgb(colour3), (x - 1/2) * 2)

def shade(data):
    sun=(-0.157991, -0.789953, -0.592464)




if __name__ == '__main__':
    data=openfile()
    #img=plt.imshow(data,cmap='nipy_spectral',clim=(-80,170), interpolation='nearest')
    #print(img)
    #plt.colorbar()

    normalcalculator(data)
    data = normalizer(data)
    plt.imshow(data)
    plt.show()