from __future__ import division
from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure,measure,feature
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray,gray2rgb
from skimage.filters import *
from matplotlib import pylab as plt
from skimage.morphology import watershed
import scipy as sci
import numpy as np
from scipy import ndimage as ndi
from numpy import array
from skimage.measure import label
from skimage import data, util
io.use_plugin('matplotlib')

nazwy=["samolot0"+str(i)+".jpg" for i in range(9)]
nazwy=nazwy+["samolot"+str(i)+".jpg" for i in range(10,21)]


def imageprocessor(data):
    K=np.array([[(1,1,1),(1,1,1),(1,1,1)],
       [(1,1,1), (-2,-2,-2), (1,1,1)],
       [(1,1,1), (1,1,1), (1,1,1)]])
    K=(K)
    data=sci.ndimage.filters.convolve(data,K)
    p1, p2 = np.percentile(data, (2, 34))
    data= exposure.rescale_intensity(data, in_range=(p1, p2))
    data =np.array([[(x[2]) for x in array] for array in data])
    data = gamma_filter(data)
    #data2 = filters.sobel(data)
    data=cutting(data)
    for i in range(int(data.shape[1]/120)):
        data = mp.erosion(data)
    return data

def cutting(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            if (data[i][j] < 0.35):
                data[i][j]=data[i][j]/2
            if (data[i][j] > 0.35):
                data[i][j] = data[i][j]*2
            if(data[i][j]>1):
                data[i][j] =1
    return data


def normalize(data):
    temparray=[]
    for array in data:
        temparray.append(max(array))
    return data/(max(array))

def imagecompiler(data1,data2):
    data1=rgb2hsv(data1)
    for i in range(len(data1)):
        for j in range(len(data1[0])):
            if(data2[i][j]>0.7):
                data1[i][j][0]=0
                data1[i][j][1]=data2[i][j]
                data1[i][j][2] = data1[i][j][2]+data2[i][j]
    return data1

def gamma_filter(img):
    MIN = 0.4628
    MAX = 0.9746
    gamma_MIN = 0.2
    gamma_MAX = 0.7
    mean = np.mean(img)
    gamma = (mean-MIN)*((gamma_MAX - gamma_MIN)/(MAX-MIN)) + gamma_MIN
    return img**gamma

if __name__ == '__main__':
    finaldata=[]
    columns = 2
    rows = 11
    fig = plt.figure(figsize=(20, 80))
    i = 1
    for nazwa in nazwy:
        print(nazwa)
        data = io.imread(nazwa)
        data = img_as_float(data)
        #data = imagecompiler(data, imageprocessor(data))
        #data=hsv2rgb(data)
        data_processed=imageprocessor(data)
        contours = measure.find_contours(data_processed, 0.7)
        ax=fig.add_subplot(rows, columns, i)
        plt.imshow(data_processed)
        ax.axis('image')
        for n, contour in enumerate(contours):
            centroid = np.sum(contour, axis=0)/len(contour)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            ax.plot(centroid[1], centroid[0], marker='o', markersize=5, color="white")
        i=i+1
    io.show()