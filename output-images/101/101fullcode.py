import numpy as np
import cv2
import os
from PIL import Image
import time

def showimage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


size=720,480

# image_name = "image104.jpg"
# im = Image.open(image_name)
# im.thumbnail(size, Image.ANTIALIAS)
# im.save("image101_new", "JPEG")
#

image_name = "image101_new"
image = cv2.imread(image_name)
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pos1 = np.where(imagegray > 128)
imagegray[pos1] = 255
pos2 = np.where(imagegray > 80) and np.where(imagegray < 120)
imagegray[pos2] = 128
pos3 = np.where(imagegray == 255)
imagegray[pos3] = 0
showimage("", imagegray)

p = np.where(imagegray == 128)
a = p[0]
b = p[1]
for i in range(len(p[0])-1):
    if imagegray[a[i]][b[i] - 1] < 10 and imagegray[a[i]][b[i] - 2] < 10 and imagegray[a[i]][b[i] - 3] < 10 and imagegray[a[i]][b[i] - 4] < 10 and imagegray[a[i]][b[i] - 5] > 10:
        if imagegray[a[i] - 1][b[i]] > 10 and imagegray[a[i] - 2][b[i]] > 10 and imagegray[a[i] - 3][b[i]] > 10:
            image[a[i]][b[i]] = [255, 255, 0]
            image[a[i]][b[i] - 1] = [255, 255, 0]
            image[a[i]][b[i] - 2] = [255, 255, 0]
            image[a[i]][b[i] - 3] = [255, 255, 0]
            image[a[i]][b[i] - 4] = [255, 255, 0]

cv2.imwrite("image_result101.jpg",image)

