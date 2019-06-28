import os
import cv2
import numpy as np
import random
import crop
from PIL import Image

def showimage(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def togray(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

def log(image):
    gray=togray(image) #gray function
    imglog=np.uint8(np.log1p(gray))
    thresh=1
    image=cv2.threshold(imglog,thresh,255,cv2.THRESH_BINARY)[1]
    return image    
    
def tophatdiffblackhat(image):
    kernel=np.ones((5,5),np.uint8)
    tophat=cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    IMAGE=blackhat-tophat
    # finalimage=tophat-blackhat
    # pos1=np.where(finalimage<[110,110,110])
    # pos2=np.where(finalimage>[130,130,130])
    # finalimage[pos1]=0
    # finalimage[pos2]=0
    showimage("",IMAGE)


if __name__=="__main__":
    image=cv2.imread("image104.jpg")
    crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=340, centre_y1=-50, centre_x2=340, centre_y2=-10,radius1=160, radius2=350, theta1=215, theta2=325)
    showimage("",crop1)
    logged=log(crop1)
    cv2.imwrite("log2.jpg",logged)
    notlogged=cv2.bitwise_not(logged)
    # tophatdiffblackhat(notlogged)
    kernel=np.array(([0,1,0],[0,1,0],[0,1,0]),dtype=np.uint8)
    erosion_8=cv2.erode(notlogged,kernel,iterations=1)
    showimage("",erosion_8)
    cv2.imwrite("image104output.jpg",erosion_8)
