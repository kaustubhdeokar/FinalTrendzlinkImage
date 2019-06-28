import numpy as np
import cv2
import os

import time

import crop
import sys
from PIL import Image


def mean(x):
    return sum(x) / len(x)


def showimage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grouping(img):
    print('the grouping part')
    image = cv2.imread(img)
    pos1 = np.where(image == [255, 0, 0])
    prev_len_of_img = len(pos1[0])
    print("original image blue pixels count",prev_len_of_img)

    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pos1 = np.where(imagegray > 128)
    imagegray[pos1] = 255
    pos2 = np.where(imagegray > 80) and np.where(imagegray < 120)
    imagegray[pos2] = 128
    pos3 = np.where(imagegray == 255)
    imagegray[pos3] = 0

    p = np.where(imagegray > 120)
    a = list(filter(lambda x: x < 600, p[0]))
    l = len(a)
    b = p[1]
    b = b[:len(a)]


    for i in range(len(a) - 3):
        x = a[i]
        y = b[i]
        try:
            for i in range(1, 5):
                for j in range(1, 5):
                    if (imagegray[x + i][y] < 10 and imagegray[x - i][y] < 10 and imagegray[x][y + i] < 10 and
                                imagegray[x][y - i] < 10 and imagegray[x - i][y - j] < 10 and imagegray[x + i][
                            y + j] < 10 and imagegray[x - i][y + j] < 10 and imagegray[x + i][y - j] < 10) :
                        image[x][y - 1] = [255, 0, 0]
                        image[x][y - 2] = [255, 0, 0]
                        image[x][y - 3] = [255, 0, 0]
                        image[x][y - 4] = [255, 0, 0]
        except:
            pass
    return image, prev_len_of_img


def gradient(image):
    a = []
    b = []
    c = []
    f = []
    for i in range(len(image) - 2):
        for j in range(len(image[i]) - 2):
            if (all(image[i][j] == 0)):
                continue
            if (abs(mean(image[i][j]) - mean(image[i][j - 1]))) > 50:  # vertical
                # print("",abs(mean(image[i][j]) - mean(image[i][j - 1])))
                a.append((i, j))

            if (abs(mean(image[i][j]) - mean(image[i - 1][j]))) > 50:  # vertical
                b.append((i, j))

            if (abs(mean(image[i][j]) - mean(image[i][j + 1]))) > 50:  # vertical
                c.append((i, j))

            # if (abs(mean(image[i][j]) - mean(image[i + 1][j]))) > 50:  # vertical
            #     f.append((i, j))

            # if (abs(mean(image[i][j]) - mean(image[i - 1][j - 1]))) > 50:  # vertical
            #     # print ('', abs(mean(image[i][j]) - mean(image[i + 1][j + 1])))
            #     f.append((i, j))


            d = set(a).intersection(set(b))
            e = set(d).intersection(set(c))
            # e = set(g).intersection(set(f))
    for i in e:
        image[i[0]][i[1]] = [255, 0, 0]
    showimage("", image)
    step1 = image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # showimage('',image_gray)
    pos1 = np.where(image_gray >= 0)
    image_gray[pos1] = 255
    positionblue = np.where(step1 == [255, 0, 0])
    a = positionblue[0]
    b = positionblue[1]
    for i in e:
        image_gray[i[0]][i[1]] = 128
    showimage('image_gray', image_gray)
    return image_gray



if __name__ == "__main__":
    start_time = time.time()

    path = os.getcwd()
    images_path_largedent = os.chdir('./largedentimages/')
    arr=[]
    for i in os.listdir('.'):
        arr.append(i)

    for i in arr:
        if i.endswith(".jpg"):
            image_name=i
            print(i)
            fname,fext=os.path.splitext(image_name)
            image = cv2.imread(image_name)
            crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=340, centre_y1=-50, centre_x2=340,
                                                        centre_y2=-10,radius1=160, radius2=350, theta1=215, theta2=325)
            # blur = cv2.medianBlur(image,5)
            # blur = ndimage.gaussian_filter(crop1, sigma=3)
            # showimage('',blur)


            path = os.getcwd()
            images_path_output = os.chdir('./output/')

            image_cropped_name = fname + "cropped.jpg"
            cv2.imwrite(image_cropped_name, crop1)

            img = cv2.imread(image_cropped_name)
            crop_img = img[190:530, 700:1300]
            # showimage('', crop_img)

            print("gradient start")
            imagegrayed = gradient(crop1)
            # showimage('', imagegrayed)

            image1 = fname + '_gray_image.jpg'

            cv2.imwrite(image1, imagegrayed)
            """
            print("groupping start")
            finalimage, pixels_cnt = grouping(image1)
            final = fname + '_final_image.jpg'

            cv2.imwrite(final, finalimage)

            # image_final = cv2.imread(final)
            # showimage("final_image", image)
            pos1 = np.where(finalimage == [255, 0, 0])
            print('Final length of image',len(pos1[0]))

            sec_pixels_cound = len(pos1[0])
            final_pixels_cnt = sec_pixels_cound - pixels_cnt
            if final_pixels_cnt == 0:
                print ("correct Part", final_pixels_cnt)

            else:
                print ("Faulty Part", final_pixels_cnt)

            path = os.getcwd()
            images_path_largedent1 = os.chdir('..')
            """
            print("--- %s seconds ---" % (time.time() - start_time))
