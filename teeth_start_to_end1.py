import numpy as np
import argparse  # parse line args
import glob
import cv2
import os
import re
from matplotlib import pyplot as plt
from PIL import Image
import crop

def mean(x):
    return sum(x)/len(x)

def showimage(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grouping(image):
    size = 1080, 720
    image_name = image
    im = Image.open(image_name)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save("tape_image", "JPEG")

    image_name = "tape_image"
    image = cv2.imread(image_name)
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pos1 = np.where(imagegray > 128)
    imagegray[pos1] = 255
    pos2 = np.where(imagegray > 80) and np.where(imagegray < 120)
    imagegray[pos2] = 128
    pos3 = np.where(imagegray == 255)
    imagegray[pos3] = 0

    cv2.imwrite("intermediate.jpg", imagegray)
    p = np.where(imagegray < 28)
    a = list(filter(lambda x: x < 600, p[0]))
    l = len(a)
    b = p[1]
    b = b[:len(a)]

    for i in range(len(a) - 1):
        x = a[i]
        y = b[i]
        temp1, temp2, temp3, temp5, temp4 = 0, 0, 0, 0, 0
        for i in range(1, 3):
            if (imagegray[x][y - i] < 10):
                temp1 = 1
            if imagegray[x - i][y - i] < 10:
                temp2 = 1
            if imagegray[x - i - 1][y - 1] > 10 and imagegray[x - i - 1][y - i - 1] > 10:
                temp3 = 1
            if imagegray[x + i][y] < 10:
                temp4 = 1
            if imagegray[x - 1][y - 1] < 10 and imagegray[x - 2][y - 1] > 10:
                temp5 = 1
            if temp1 == temp2 == temp3 == temp4 == temp5:
                image[x][y - 1] = [255, 255, 0]
                image[x][y - 2] = [255, 255, 0]
                image[x][y - 3] = [255, 255, 0]
                image[x][y - 4] = [255, 255, 0]

    cv2.imwrite("tape_imageresults.jpg", image)
    showimage("", image)

def gradient(imagename):
    image = cv2.imread(imagename)
    a=[]
    b=[]
    for i in range(len(image)-2):
        for j in range(len(image[i])-2):
            if (mean(image[i][j]) - mean(image[i+1][j])) > 50:  # vertical
                a.append((i,j))
    
    for i in range(len(image)-2):
        for j in range(len(image[i])-2):
            if (mean(image[i][j]) - mean(image[i][j+1])) > 50:  # vertical
                b.append((i,j))
    c=set(a).intersection(set(b))
    for i in c:
        image[i[0]][i[1]]=[0,255,255]

    step1=image
    cv2.imwrite("step1.jpg",step1)
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    pos1=np.where(image_gray>=0)
    image_gray[pos1]=255
    positionblue=np.where(step1==[0,255,255])
    a=positionblue[0]
    b=positionblue[1]
    for i in range(len(a)):
        image_gray[a[i]][b[i]]=0
    cv2.imwrite("image_gray.jpg",image_gray)
    return image_gray



image_name = "img6_1.jpg"
fname, fext = os.path.splitext(image_name)
image=cv2.imread(image_name)
imagegrayed=gradient(image_name)
showimage("",imagegrayed)
grouping("image_gray.jpg")

