import cv2
import numpy as np
import crop
import time


def showimage(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tophatdiffblackhat(image): 
    kernel=np.ones((5,5),np.uint8)
    tophat=cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    finalimage=tophat-blackhat
    showimage("",finalimage)
    

if __name__=="__main__":
    st=time.time()
    image=cv2.imread("image106.jpg")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    showimage("",image)
    tophatdiffblackhat(image)
    # pos1=np.where(image[:,:,0]==image[:,:,1])
    # pos2=np.where(image[:,:,1]==image[:,:,2])
    # pos1set=[]
    # pos2set=[]
    # for i in range(len(pos1[0])):
    #     pos1set.append((pos1[0][i],pos1[1][i]))
    # for i in range(len(pos2[0])):
    #     pos2set.append((pos2[0][i],pos2[1][i]))
    
    # c=list(set(pos1set).intersection(set(pos2set)))
    
    # img=np.ones((image.shape[0],image.shape[1]))
    
    # for i in range(len(c)):
    #     img[c[i][0]][c[i][1]]=0
    
    # cv2.imshow("image",image)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)