import numpy as np
import cv2
import os
from PIL import Image
import time


starttime=time.time()
size=720,480

image_name = "image105.jpg"
im = Image.open(image_name)
im.thumbnail(size, Image.ANTIALIAS)
im.save("image105_new", "JPEG")

image_name = "image105_new"
image = cv2.imread(image_name)
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pos1 = np.where(imagegray > 128) and np.where(imagegray<240)
imagegray[pos1] = 128
pos2 = np.where(imagegray > 240)
imagegray[pos2] = 0

p = np.where(imagegray == 128)
a = p[0]
b = p[1]

for i in range(len(p[0])-1):
    if imagegray[a[i]][b[i] - 4] < 10 and imagegray[a[i]][b[i] - 3] < 10 and imagegray[a[i]][b[i] - 1] < 10 and imagegray[a[i]][b[i] - 1] < 10 and imagegray[a[i]][b[i] - 2] < 10 :
        if imagegray[a[i] - 1][b[i]] > 10 and imagegray[a[i] - 2][b[i]] > 10 and imagegray[a[i] - 3][b[i]] > 10:
            image[a[i]][b[i]] = [255, 255, 0]
            image[a[i]][b[i] - 1] = [255, 255, 0]
            image[a[i]][b[i] - 2] = [255, 255, 0]
            image[a[i]][b[i] - 3] = [255, 255, 0]
            image[a[i]][b[i] - 4] = [255, 255, 0]
endtime=time.time()
print("time required is {}".format(endtime-starttime))
cv2.imwrite("image105result.jpg",image)

