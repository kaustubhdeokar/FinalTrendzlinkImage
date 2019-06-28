import numpy as np
import argparse #parse line args
import glob
import cv2
import os
import re
from matplotlib import pyplot as plt
from PIL import Image
from envision import crop

from convolution import *

kernel1 = np.array((
[1, 2, 1],
[0, 0, 0],
[-1, -2, -1]), dtype="int")

kernel2 = np.array((
[1, 0, -1],
[2, 0, -2],
[1, 0, -1]), dtype="int")

kernel3 = np.array((
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]), dtype="int")

kernel4 = np.array((
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]), dtype="int")

threshold_val = 50

def convolveimage():
    image_1 = np.array(Image.open("/media/pooja/G-drive/Trendzlink/projects/Aligrow/images/cam3_1.jpg"))
    gray_image = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    cv2.imshow("", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    convolution1 = (signal.convolve2d(gray_image, kernel1, boundary="symm", mode="valid", fillvalue=0))
    im1 = convolution1
    # test = np.zeros(im1.shape,dtype="int")

    convolution2 = (signal.convolve2d(gray_image, kernel2, boundary="symm", mode="valid", fillvalue=0))
    im2 = convolution2

    convolution3 = (signal.convolve2d(gray_image, kernel3, boundary="symm", mode="valid", fillvalue=0))
    im3 = convolution3

    convolution4 = (signal.convolve2d(gray_image, kernel4, boundary="symm", mode="valid", fillvalue=0))
    im4 = convolution4


    pos1 = np.where(im1 > threshold_val)
    pos2 = np.where(im2 > threshold_val)
    pos3 = np.where(im3 > threshold_val)
    pos4 = np.where(im4 > threshold_val)

    test = np.zeros(gray_image.shape,dtype="int")
    cv2.imwrite("scharr_1.jpg", test)
    test[pos1] = 255
    test[pos2] = 255
    test[pos3] = 255
    test[pos4] = 255

    cv2.imshow("", test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




convolveimage()