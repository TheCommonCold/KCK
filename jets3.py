from __future__ import division
from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray,gray2rgb
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array

def imageprocessor(data):
    data = rgb2gray(data)
    data = (filters.sobel(data))
    data= mp.dilation(data)
    data=mp.dilation(data)
    return data

def imagecompiler(data1,data2):
    data1=rgb2hsv(data1)
    for i in range(len(data1)):
        for j in range(len(data1[0])):
            if(data2[i][j]>0.1):
                data1[i][j][0]=0
                data1[i][j][1]=1
                data1[i][j][2] = data1[i][j][2]+data2[i][j]
    return hsv2rgb(data1)

if __name__ == '__main__':
    data=io.imread("samolot11.jpg")
    data = img_as_float(data)
    data=imagecompiler(data,imageprocessor(data))
    io.imshow(data)
    io.show()