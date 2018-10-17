#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import colorsys

from matplotlib import colors

def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    #rc('text', usetex=True)
    #rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400         # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)


    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')
    fig.show()

def interpolator(rgb1,rgb2,t):

    return rgb1[0] + (rgb2[0] - rgb1[0]) * t,rgb1[1] + (rgb2[1] - rgb1[1]) * t,rgb1[2] + (rgb2[2] - rgb1[2]) * t


def gradient_rgb_bw(v):

    return (v, v, v)


def gradient_rgb_gbr(v):
    rgb1=[0,1,0]
    rgb2=[0,0,1]
    rgb3=[1,0,0]
    if(v<0.5):
        return interpolator(rgb1,rgb2,v*2)
    else:
        return interpolator(rgb2, rgb3, (v-0.5) * 2)


def gradient_rgb_gbr_full(v):
    rgb1 = [0, 1, 0]
    rgb2 = [0, 1, 1]
    rgb3 = [0, 0, 1]
    rgb4 = [1, 0, 1]
    rgb5 = [1, 0, 0]
    if(v<0.25):
        return interpolator(rgb1,rgb2,v*4)
    elif(v<0.5):
        return interpolator(rgb2, rgb3, (v-0.25) * 4)
    elif (v < 0.75):
        return interpolator(rgb3, rgb4, (v - 0.5) * 4)
    else:
        return interpolator(rgb4, rgb5, (v - 0.75) * 4)


def gradient_rgb_wb_custom(v):
    rgb1 = [0, 0, 0]
    rgb2 = [0.5, 1, 1]
    rgb3 = [0, 0, 0]
    rgb4 = [1, 0, 0.6]
    rgb5 = [0, 1, 0]
    rgb6 = [1, 0.3, 1]
    rgb7 = [0.2, 0, 1]
    rgb8 = [1, 1, 1]
    if(v<0.125):
        return interpolator(rgb1,rgb2,v*(1/0.125))
    elif(v<0.25):
        return interpolator(rgb2, rgb3, (v-0.125)*(1/0.125))
    elif (v < 0.375):
        return interpolator(rgb3, rgb4, (v-0.25)*(1/0.125))
    elif(v<0.5):
        return interpolator(rgb4, rgb5, (v-0.375)*(1/0.125))
    elif(v<0.625):
        return interpolator(rgb5, rgb6, (v-0.5)*(1/0.125))
    elif(v<0.75):
        return interpolator(rgb6, rgb7, (v-0.625)*(1/0.125))
    elif(v<0.875):
        return interpolator(rgb7, rgb1, (v-0.75)*(1/0.125))
    else:
        return interpolator(rgb1, rgb8, (v-0.875)*(1/0.125))

def gradient_hsv_bw(v):
    return (interpolator(colorsys.hsv_to_rgb(0,0,0),colorsys.hsv_to_rgb(0,0,1),v))



def gradient_hsv_gbr(v):
    rgb1 = colorsys.hsv_to_rgb(1/3, 1, 1)
    rgb2 = colorsys.hsv_to_rgb(1/2, 1, 1)
    rgb3 = colorsys.hsv_to_rgb(2/3, 1, 1)
    rgb4 = colorsys.hsv_to_rgb(5/6, 1, 1)
    rgb5 = colorsys.hsv_to_rgb(0, 1, 1)
    if(v<0.25):
        return interpolator(rgb1,rgb2,v*4)
    elif(v<0.5):
        return interpolator(rgb2, rgb3, (v-0.25) * 4)
    elif (v < 0.75):
        return interpolator(rgb3, rgb4, (v - 0.5) * 4)
    else:
        return interpolator(rgb4, rgb5, (v - 0.75) * 4)
def gradient_hsv_unknown(v):
    rgb1=colorsys.hsv_to_rgb(120/255,0.8,1)
    rgb2=colorsys.hsv_to_rgb(45/255,0.8,1)
    rgb3=colorsys.hsv_to_rgb(1/255,0.8,1)
    if(v<0.5):
        return interpolator(rgb1,rgb2,v*2)
    else:
        return interpolator(rgb2, rgb3, (v-0.5) * 2)


def gradient_hsv_custom(v):
    rgb1 = colorsys.hsv_to_rgb(1/3, 1, 0)
    rgb2 = colorsys.hsv_to_rgb(1/9, 1, 0.75)
    rgb3 = colorsys.hsv_to_rgb(1/2, 0.4, 0.8)
    rgb4 = colorsys.hsv_to_rgb(0.25, 1, 1)
    rgb5 = colorsys.hsv_to_rgb(0.89, 0.2, 1)
    rgb6 = colorsys.hsv_to_rgb(0.457, 1, 0.5)
    rgb7 = colorsys.hsv_to_rgb(0.612, .8, 1)
    rgb8 = colorsys.hsv_to_rgb(1/3, 0, 1)
    if (v < 0.125):
        return interpolator(rgb1, rgb2, v * (1 / 0.125))
    elif (v < 0.25):
        return interpolator(rgb2, rgb3, (v - 0.125) * (1 / 0.125))
    elif (v < 0.375):
        return interpolator(rgb3, rgb4, (v - 0.25) * (1 / 0.125))
    elif (v < 0.5):
        return interpolator(rgb4, rgb5, (v - 0.375) * (1 / 0.125))
    elif (v < 0.625):
        return interpolator(rgb5, rgb6, (v - 0.5) * (1 / 0.125))
    elif (v < 0.75):
        return interpolator(rgb6, rgb7, (v - 0.625) * (1 / 0.125))
    elif (v < 0.875):
        return interpolator(rgb7, rgb1, (v - 0.75) * (1 / 0.125))
    else:
        return interpolator(rgb1, rgb8, (v - 0.875)*(1/0.125))

if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])