import cv2
import numpy as np

def showimage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('tape_imageresults.jpg')
showimage('',image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
showimage('',image)
pos1 = np.where(image==[255,255,0])
image[pos1]= 255

showimage('',image)
