import numpy as np
import cv2
import os
from PIL import Image
import crop


def thresholding(image,threshold=40):


    ret,mask=cv2.threshold(image,threshold,100,cv2.THRESH_BINARY_INV)
    return mask

def gaps_frontal(image_name):
    #full name
    image_read=image_name+".jpg"

    image=cv2.imread(image_read,0)
    crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=345, centre_y1=-105, centre_x2=340, centre_y2=-65,
                                                   radius1=345, radius2=380, theta1=210, theta2=330)
    ret,thresh1 = cv2.threshold(crop1,185,205,cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(thresh1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,13)
    res=cv2.bitwise_and(thresh1,th3)
    pos1=np.where(res>100)
    res[pos1]=255
    save_as="gaps_between_frontal_"+image_read
    cv2.imwrite(save_as,res)
    
def full_teeth(image_name):
    #full tooth
    image_read=image_name+".jpg"
    image=cv2.imread(image_read,0)
    crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=330, centre_y1=-115, centre_x2=340, centre_y2=-30,
                        radius1=215, radius2=330, theta1=210, theta2=330)
    
    
    ret,thresh1 = cv2.threshold(crop1,190,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(crop1,165,255,cv2.THRESH_BINARY)

    
    kernel = np.ones((5,5), np.uint8) 
    #image_dilated=cv2.erode(thresh1,kernel,iterations=3)
    save_as="full_teeth_"+image_read
    save_as_for_portion_between_tooth="full_teeth_for_portion_between_tooth_"+image_read
    cv2.imwrite(save_as,thresh1)
    cv2.imwrite(save_as_for_portion_between_tooth,thresh2)

    
def results_only_teeth(image_name):
    #to show only the teeth
    gaps="gaps_between_frontal_"+image_name+".jpg"
    full_teeth="full_teeth_"+image_name+".jpg"
    img1=cv2.imread(gaps,0)
    img2=cv2.imread(full_teeth,0)
    img4=img2-img1
    kernel = np.ones((3,3), np.uint8) 
    img5=cv2.erode(img4,kernel,iterations=1)
    save_as="only_teeth_"+image_name+".jpg"
    cv2.imwrite(save_as,img5)

def lower_tooth(image_name):
    image_read=image_name+".jpg"
    image=cv2.imread(image_read)
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #-45
    crop1, _ = crop.crop_radial_arc_two_centres(image_gray, centre_x1=330, centre_y1=-130, centre_x2=330, centre_y2=-125 ,
             radius1=355, radius2=440, theta1=210, theta2=330)

    cv2.imshow("", crop1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    convert=np.where(crop1<10)
    crop1[convert]=128
    pos1=np.where(crop1<70)
    crop1[pos1]=0
    pos1=np.where(crop1>70)
    crop1[pos1]=255

    kernel = np.ones((3, 3), np.uint8)
    img5 = cv2.erode(crop1, kernel, iterations=2)

    save_as="lower_tooth_"+image_read
    cv2.imwrite(save_as,img5)
    cv2.imshow("", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save_as="lower_tooth_"+image_read
    # cv2.imwrite(save_as,crop1)


def portion_between_tooth(image_name):
    image_read=image_name+".jpg"
    image=cv2.imread(image_read)
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    pos1=np.where(image_gray<100)
    image_gray[pos1]=0
    pos2=np.where(image_gray>100)
    image_gray[pos2]=255

    # pos3=np.where(image_gray>100)
    # image_gray[pos3]=255

 
    crop1, _ = crop.crop_radial_arc_two_centres(image_gray, centre_x1=330, centre_y1=-115, centre_x2=340, centre_y2=-30,
                        radius1=215, radius2=330, theta1=210, theta2=330)
    kernel = np.ones((5,5), np.uint8) 
    img5=cv2.erode(crop1,kernel,iterations=1)
    
    save_as="portion_between_tooth_croppped_"+image_read
    cv2.imwrite(save_as,crop1)

    to_read_a="only_teeth_"+image_read
    to_read_b=save_as
    a=cv2.imread(to_read_a)
    print(a)
    b=cv2.imread(to_read_b)
    print(b)
    
    c=b-a
    kernel = np.ones((5,5), np.uint8) 
    img5=cv2.erode(c,kernel,iterations=1)

    # img6=cv2.dilate(img5,kernel,iterations=2)
    save_as="portion-between-tooth_"+image_read
    # img8=cv2.cvtColor(img6,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_as, img5)
    cv2.imshow("",img5)

    img6=cv2.dilate(img5,kernel,iterations=1)
    save_as="portion_between_tooth_"+image_read

    cv2.imshow("",img6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def remaining_portion(fname):
    image_read=fname+".jpg"
    image=cv2.imread(image_read)
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    image_full_teet=cv2.imread("portion_between_tooth_croppped_213_1.jpg")
    image_full_teeth=cv2.cvtColor(image_full_teet,cv2.COLOR_BGR2GRAY)

    pos=np.where(image_full_teeth==255)
    image_gray[pos]=255

    image_lower_teet=cv2.imread("lower_tooth_213_1.jpg")
    image_lower_teeth=cv2.cvtColor(image_lower_teet,cv2.COLOR_BGR2GRAY)
    image_lower_teeth_inv=cv2.bitwise_not(image_lower_teeth)

    pos2=np.where(image_lower_teeth==0)
    image_gray[pos2]=255
    
    pos3=np.where(image_gray<100)
    image_gray[pos3]=0

    pos4=np.where(image_gray>10)
    image_gray[pos4]=255

    pos5=np.where(image_gray<10)
    image_gray[pos5]=128
    

    crop1, _ = crop.crop_radial_arc_two_centres(image_gray, centre_x1=330, centre_y1=-115, centre_x2=340, centre_y2=-30,
                        radius1=215, radius2=330, theta1=210, theta2=330)
    final_image=cv2.bitwise_not(crop1)
    cv2.imwrite("side_tooth.jpg",final_image)




image_name="213_1.jpg"
fname,fext=os.path.splitext(image_name)
gaps_frontal(fname)
full_teeth(fname)
lower_tooth(fname)
results_only_teeth(fname)
lower_tooth(fname)
portion_between_tooth(fname)
remaining_portion(fname)


