import cv2
import numpy as np
import crop


def crop_image():
    image = cv2.imread('img6_1.jpg')
    crop1, _ = crop.crop_radial_arc_two_centres(image, centre_x1=325, centre_y1=-45, centre_x2=340, centre_y2=-30,
                                                radius1=215, radius2=470, theta1=210, theta2=330)
    cv2.imwrite('crop1.jpg', crop1)

crop_image()


