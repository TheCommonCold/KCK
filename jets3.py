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
from skimage.filters import *
from matplotlib import pylab as plt
import numpy as np
from numpy import array
io.use_plugin('matplotlib')

nazwy=["samolot0"+str(i)+".jpg" for i in range(9)]
#nazwy=nazwy+["samolot"+str(i)+".jpg" for i in range(10,21)]

def imageprocessor(data):
    data =[[(x[0]) for x in array] for array in data]
    K=np.array([[1,1,1],
       [1, 1, 1],
       [1, 1, 1]])
    K=K/10
    data=filters.convolve(data,K)
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
    return data1


if __name__ == '__main__':
    finaldata=[]
    columns = 3
    rows = 3
    fig = plt.figure(figsize=(20, 20))
    i = 1
    for nazwa in nazwy:
        print(nazwa)
        data = io.imread(nazwa)
        data = img_as_float(data)
        #data = imagecompiler(data, imageprocessor(data))
        data=imageprocessor(data)
        #data=matplotlib.colors.hsv_to_rgb(data)
        fig.add_subplot(rows, columns, i)
        plt.imshow(data)
        i=i+1
    io.show()