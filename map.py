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

def normalizer(data):
    temp=[]
    for array in data:
        temp.append(min(array))
        temp.append(max(array))
    minimum=min(temp)
    maximum=max(temp)
    data=[[(x-minimum)/(maximum-minimum)for x in array] for array in data]
    return data
def gradient(x):
    colour1=(10/30,1,0.8)
    colour2=(6/36,1,1)
    colour3=(0,1,1)
    if (x<1/2):
        return interpolator(hsv_to_rgb(colour1), hsv_to_rgb(colour2), (x*2))
    else:
        return interpolator(hsv_to_rgb(colour2), hsv_to_rgb(colour3), (x - 1/2) * 2)


def imgtogradient(data):
    img=[[gradient(x)for x in array] for array in data]
    #print(img)
    return img




if __name__ == '__main__':
    data=openfile()
    #img=plt.imshow(data,cmap='nipy_spectral',clim=(-80,170), interpolation='nearest')
    #print(img)
    #plt.colorbar()
    data = normalizer(data)
    img = imgtogradient(data)
    plt.imshow(img)
    plt.show()