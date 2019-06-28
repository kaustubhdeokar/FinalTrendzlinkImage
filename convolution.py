import numpy as np
from scipy import signal
import cv2
from envision.crop import *

""" SOBEL KERNELS """

"""
8 kernels sobel
SOBEL_KERNEL_LEFT_RIGHT
SOBEL_KERNEL_RIGHT_LEFT
SOBEL_KERNEL_TOP_BOTTOM
SOBEL_KERNEL_BOTTOM_TOP
SOBEL_KERNEL_DIAGONAL_TOP_LEFT
SOBEL_KERNEL_DIAGONAL_BOTTOM_LEFT
SOBEL_KERNEL_DIAGONAL_TOP_RIGHT
SOBEL_KERNEL_DIAGONAL_BOTTOM_RIGHT
"""

SOBEL_KERNEL_LEFT_RIGHT = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")

SOBEL_KERNEL_RIGHT_LEFT = np.array((
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]), dtype="int")

SOBEL_KERNEL_TOP_BOTTOM = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

SOBEL_KERNEL_BOTTOM_TOP = np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype="int")

SOBEL_KERNEL_DIAGONAL_TOP_LEFT = np.array((
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]), dtype="int")

SOBEL_KERNEL_DIAGONAL_BOTTOM_LEFT = np.array((
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]), dtype="int")

SOBEL_KERNEL_DIAGONAL_TOP_RIGHT = np.array((
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]), dtype="int")

SOBEL_KERNEL_DIAGONAL_BOTTOM_RIGHT = np.array((
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]), dtype="int")

""" SCHARR KERNELS """
"""
8 kernels scharr
SCHARR_KERNEL_LEFT_RIGHT
SCHARR_KERNEL_RIGHT_LEFT
SCHARR_KERNEL_TOP_BOTTOM
SCHARR_KERNEL_BOTTOM_TOP
SCHARR_KERNEL_DIAGONAL_TOP_LEFT
SCHARR_KERNEL_DIAGONAL_BOTTOM_LEFT
SCHARR_KERNEL_DIAGONAL_TOP_RIGHT
SCHARR_KERNEL_DIAGONAL_BOTTOM_RIGHT
"""
SCHARR_KERNEL_LEFT_RIGHT = np.array((
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]), dtype="int")

SCHARR_KERNEL_RIGHT_LEFT = np.array((
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]), dtype="int")

SCHARR_KERNEL_TOP_BOTTOM = np.array((
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3]), dtype="int")

SCHARR_KERNEL_BOTTOM_TOP = np.array((
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]), dtype="int")

SCHARR_KERNEL_DIAGONAL_TOP_LEFT = np.array((
        [-10, -3, 0],
        [-3, 0, 3],
        [0, 3, 10]), dtype="int")

SCHARR_KERNEL_DIAGONAL_BOTTOM_LEFT = np.array((
        [0, 3, 10],
        [-3, 0, 3],
        [-10, -3, 0]), dtype="int")

SCHARR_KERNEL_DIAGONAL_TOP_RIGHT = np.array((
        [0, -3, -10],
        [3, 0, -3],
        [10, 3, 0]), dtype="int")

SCHARR_KERNEL_DIAGONAL_BOTTOM_RIGHT = np.array((
        [0, -3, -10],
        [3, 0, -3],
        [10, 3, 0]), dtype="int")


def convolve_sobel(img=None,
                   threshold=0,
                   sobel_kernel_left_right=True,
                   sobel_kernel_right_left=True,
                   sobel_kernel_top_bottom=True,
                   sobel_kernel_bottom_top=True,
                   sobel_kernel_diagonal_top_left=False,
                   sobel_kernel_diagonal_bottom_left=False,
                   sobel_kernel_diagonal_top_right=False,
                   sobel_kernel_diagonal_bottom_right=False):
    """
    Convolution of sobel
    :param img:
    :param threshold:
    :param sobel_kernel_left_right:
    :param sobel_kernel_right_left:
    :param sobel_kernel_top_bottom:
    :param sobel_kernel_bottom_top:
    :param sobel_kernel_diagonal_top_left:
    :param sobel_kernel_diagonal_bottom_left:
    :param sobel_kernel_diagonal_top_right:
    :param sobel_kernel_diagonal_bottom_right:
    :return:
    """
    test = np.zeros(img.shape, dtype="int")

    if sobel_kernel_left_right:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_LEFT_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_right_left:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_RIGHT_LEFT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_bottom_top:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_BOTTOM_TOP, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_top_bottom:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_TOP_BOTTOM, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_diagonal_bottom_left:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_DIAGONAL_BOTTOM_LEFT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_diagonal_top_left:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_DIAGONAL_TOP_LEFT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_diagonal_bottom_right:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_DIAGONAL_BOTTOM_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if sobel_kernel_diagonal_top_right:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_DIAGONAL_TOP_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    return test


def convolve_scharr(img=None,
                    threshold=0,
                    scharr_kernel_left_right=True,
                    scharr_kernel_right_left=True,
                    scharr_kernel_top_bottom=True,
                    scharr_kernel_bottom_top=True,
                    scharr_kernel_diagonal_top_left=False,
                    scharr_kernel_diagonal_bottom_left=False,
                    scharr_kernel_diagonal_top_right=False,
                    scharr_kernel_diagonal_bottom_right=False):

    test = np.zeros(img.shape, dtype="int")

    if scharr_kernel_left_right:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_LEFT_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_right_left:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_RIGHT_LEFT, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_bottom_top:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_BOTTOM_TOP, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_top_bottom:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_TOP_BOTTOM, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_diagonal_bottom_left:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_DIAGONAL_BOTTOM_LEFT,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_diagonal_top_left:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_DIAGONAL_TOP_LEFT,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_diagonal_bottom_right:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_DIAGONAL_BOTTOM_RIGHT,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if scharr_kernel_diagonal_top_right:
        convolution = (signal.convolve2d(img, SCHARR_KERNEL_DIAGONAL_TOP_RIGHT,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    return test


def convolve_custom_kernel(img=None,
                    threshold=0,
                    custom_kernel=None,
                    enable_custom_kernel_left_right=True,
                    enable_custom_kernel_right_left=True,
                    enable_custom_kernel_top_bottom=True,
                    enable_custom_kernel_bottom_top=True,
                    enable_custom_kernel_diagonal_top_left=False,
                    enable_custom_kernel_diagonal_bottom_left=False,
                    enable_custom_kernel_diagonal_top_right=False,
                    enable_custom_kernel_diagonal_bottom_right=False):

    test = np.zeros(img.shape, dtype="int")

    custom_kernel_left_right = custom_kernel
    custom_kernel_right_left = np.flip(custom_kernel, 1)
    custom_kernel_top_bottom = custom_kernel.T
    custom_kernel_bottom_top = np.flip(custom_kernel.T, 0)

    custom_kernel_diagonal_top_left = np.array((
        [custom_kernel[0][1], custom_kernel[0][0], custom_kernel[1][0]],
        [custom_kernel[0][2], custom_kernel[1][1], custom_kernel[2][0]],
        [custom_kernel[1][2], custom_kernel[2][2], custom_kernel[2][1]]), dtype="int")

    custom_kernel_diagonal_bottom_left = np.flip(custom_kernel_diagonal_top_left, 0)
    custom_kernel_diagonal_top_right = np.flip(custom_kernel_diagonal_top_left, 1)
    custom_kernel_diagonal_bottom_right = np.flip(custom_kernel_diagonal_bottom_left, 1)

    if enable_custom_kernel_left_right:
        convolution = (signal.convolve2d(img, custom_kernel_left_right, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_right_left:
        convolution = (signal.convolve2d(img, custom_kernel_right_left, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_top_bottom:
        convolution = (signal.convolve2d(img, custom_kernel_top_bottom, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_bottom_top:
        convolution = (signal.convolve2d(img, custom_kernel_bottom_top, boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_diagonal_bottom_left:
        convolution = (signal.convolve2d(img, custom_kernel_diagonal_bottom_left,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_diagonal_bottom_right:
        convolution = (signal.convolve2d(img, custom_kernel_diagonal_bottom_right,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_diagonal_top_left:
        convolution = (signal.convolve2d(img, custom_kernel_diagonal_top_left,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    if enable_custom_kernel_diagonal_top_right:
        convolution = (signal.convolve2d(img, custom_kernel_diagonal_top_right,
                                         boundary="symm", mode="valid", fillvalue=0))
        pos = np.where(convolution > threshold)
        test[pos] = 255

    return test


def convolve_sobel_multi_threshold_radial(img=None,
                                          multi_thresholds=None,
                                          centre_x=None,
                                          centre_y=None,
                                          sobel_kernel_left_right=True,
                                          sobel_kernel_right_left=True,
                                          sobel_kernel_top_bottom=True,
                                          sobel_kernel_bottom_top=True,
                                          sobel_kernel_diagonal_top_left=False,
                                          sobel_kernel_diagonal_bottom_left=False,
                                          sobel_kernel_diagonal_top_right=False,
                                          sobel_kernel_diagonal_bottom_right=False):
    """
    Apply sobel convolution radially by selecting
    arcs between theta1 and theta2

    multi_thresholds = [(theta11, theta12, threshold1), (theta21, theta22, threshold2), (theta31, theta32, threshold3)]

    eg. multi_thresholds = [(0, 50, 100), (50, 180, 200)]

    :param img:
    :param multi_thresholds:
    :param centre_x:
    :param centre_y:
    :param sobel_kernel_left_right:
    :param sobel_kernel_right_left:
    :param sobel_kernel_top_bottom:
    :param sobel_kernel_bottom_top:
    :param sobel_kernel_diagonal_top_left:
    :param sobel_kernel_diagonal_bottom_left:
    :param sobel_kernel_diagonal_top_right:
    :param sobel_kernel_diagonal_bottom_right:
    :return:
    """
    test = np.zeros(img.shape, dtype="int")

    if multi_thresholds is None:
        """ Default value of multi_thresholds theta1=0, theta2=360, threshold=255 """
        multi_thresholds.append((0, 360, 255))

    if centre_x is None:
        centre_x = img.shape[0] / 2
    if centre_y is None:
        centre_y = img.shape[1] / 2

    if sobel_kernel_left_right:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_LEFT_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_right_left:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_RIGHT_LEFT, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_bottom_top:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_BOTTOM_TOP, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_top_bottom:
        convolution = (signal.convolve2d(img, SOBEL_KERNEL_TOP_BOTTOM, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_diagonal_bottom_left:
        convolution = (signal.convolve2d
                       (img, SOBEL_KERNEL_DIAGONAL_BOTTOM_LEFT, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_diagonal_top_left:
        convolution = (signal.convolve2d
                       (img, SOBEL_KERNEL_DIAGONAL_TOP_LEFT, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_diagonal_bottom_right:
        convolution = (signal.convolve2d
                       (img, SOBEL_KERNEL_DIAGONAL_BOTTOM_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    if sobel_kernel_diagonal_top_right:
        convolution = (signal.convolve2d
                       (img, SOBEL_KERNEL_DIAGONAL_TOP_RIGHT, boundary="symm", mode="valid", fillvalue=0))
        theta = find_angle_from_given_point(convolution, centre_x, centre_y)
        for theta1, theta2, threshold in multi_thresholds:
            pos = np.where((convolution > threshold) & (theta > theta1) & (theta <= theta2))
            test[pos] = 255

    return test
