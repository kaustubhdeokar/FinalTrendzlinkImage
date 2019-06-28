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
    


if __name__=="__main__":
    image=cv2.imread("image106.jpg")
    crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=340, centre_y1=-50, centre_x2=340, centre_y2=-10,radius1=160, radius2=350, theta1=215, theta2=325)
    logged=log(crop1)
    showimage("",logged)
    cv2.imwrite("log2.jpg",logged)
    notlogged=cv2.bitwise_not(logged)
    
    kernel=np.array(([1,1,1],[0,1,0],[1,1,1]),dtype=np.uint8)
    erosion_8=cv2.filter2D()