import cv2
import numpy as np
import crop


def showimage(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tophatdiffblackhat(image):
    showimage("",image)
    kernel=np.ones((5,5),np.uint8)
    tophat=cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    finalimage=tophat-blackhat
    pos1=np.where(finalimage<[110,110,110])
    pos2=np.where(finalimage>[130,130,130])
    finalimage[pos1]=0
    finalimage[pos2]=0
    cv2.imwrite("output108.jpg",finalimage)
    
if __name__=="__main__":
    image=cv2.imread("image108.jpg")
    crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=340, centre_y1=-50, centre_x2=340, centre_y2=-10,radius1=160, radius2=350, theta1=215, theta2=325)
    tophatdiffblackhat(crop1)