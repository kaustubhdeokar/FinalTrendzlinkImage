from __future__ import division
import numpy as np


def find_angle_from_given_point(img, centre_x, centre_y):
    y_area = np.arange(0, img.shape[0], dtype=np.float)
    x_area = np.arange(0, img.shape[1], dtype=np.float)

    xx, yy = np.meshgrid(x_area, y_area)

    theta = np.arctan2(yy - centre_y, xx - centre_x) * 180 / np.pi + 180

    return theta

def find_radial_distance_from_given_point(img, centre_x, centre_y):
    y_area = np.arange(0, img.shape[0], dtype=np.int64)
    x_area = np.arange(0, img.shape[1], dtype=np.int64)

    xx, yy = np.meshgrid(x_area, y_area)

    r = np.sqrt(np.power((xx - centre_x), 2) + np.power((yy - centre_y), 2))

    return r


def crop_circle(img, centre_x=None, centre_y=None, radius=None, inside_region=True):
    test = np.copy(img)

    if centre_x is None:
        centre_x = img.shape[0] / 2
    if centre_y is None:
        centre_y = img.shape[1] / 2

    if radius is None:
        if img.shape[1] > img.shape[0]:
            radius = img.shape[0] / 2
        else:
            radius = img.shape[1] / 2

    r = find_radial_distance_from_given_point(img, centre_x, centre_y)

    if inside_region:
        unselected_area = np.where(r > radius)
        selected_area = np.where(r <= radius)
    else:
        unselected_area = np.where(r < radius)
        selected_area = np.where(r >= radius)

    test[unselected_area] = 0

    return test, test[selected_area].flatten().tolist()


def crop_ring(img, centre_x=None, centre_y=None, radius1=None, radius2=None, inside_region=True,
              unselected_area_value=0):
    test = np.copy(img)

    if centre_x is None:
        centre_x = img.shape[1] / 2
    if centre_y is None:
        centre_y = img.shape[0] / 2

    if radius1 is None:
        radius1 = 0

    if radius2 is None:
        if img.shape[1] > img.shape[0]:
            radius2 = img.shape[0] / 2
        else:
            radius2 = img.shape[1] / 2

    r = find_radial_distance_from_given_point(img, centre_x, centre_y)

    if inside_region:
        unselected_area = np.where((r < radius1) | (r > radius2))
        selected_area = np.where((r >= radius1) & (r <= radius2))
    else:
        unselected_area = np.where((r >= radius1) & (r <= radius2))
        selected_area = np.where((r < radius1) | (r > radius2))

    test[unselected_area] = unselected_area_value

    return test, test[selected_area].flatten().tolist()


def crop_ring_two_centers(img, centre_x1=None, centre_y1=None, centre_x2=None, centre_y2=None,
                          radius1=None, radius2=None, inside_region=True, unselected_area_value=0):
    test = np.copy(img)

    if centre_x1 is None:
        centre_x1 = img.shape[1] / 2
    if centre_y1 is None:
        centre_y1 = img.shape[0] / 2
    if centre_x2 is None:
        centre_x2 = img.shape[1] / 2
    if centre_y2 is None:
        centre_y2 = img.shape[0] / 2

    if radius1 is None:
        radius1 = 0

    if radius2 is None:
        if img.shape[1] > img.shape[0]:
            radius2 = img.shape[0] / 2
        else:
            radius2 = img.shape[1] / 2

    r1 = find_radial_distance_from_given_point(img, centre_x1, centre_y1)
    r2 = find_radial_distance_from_given_point(img, centre_x2, centre_y2)

    if inside_region:
        unselected_area = np.where((r1 < radius1) | (r2 > radius2))
        selected_area = np.where((r1 >= radius1) & (r2 <= radius2))
    else:
        unselected_area = np.where((r1 >= radius1) & (r2 <= radius2))
        selected_area = np.where((r1 < radius1) | (r2 > radius2))

    test[unselected_area] = unselected_area_value

    return test, test[selected_area].flatten().tolist()


def crop_radial_arc(img, centre_x=None, centre_y=None, radius1=None, radius2=None, theta1=0, theta2=360,
                    inside_region=True):
    test = np.copy(img)

    if centre_x is None:
        centre_x = img.shape[0] / 2
    if centre_y is None:
        centre_y = img.shape[1] / 2

    if radius1 is None:
        radius1 = 0

    if radius2 is None:
        if img.shape[1] > img.shape[0]:
            radius2 = img.shape[0] / 2
        else:
            radius2 = img.shape[1] / 2

    r = find_radial_distance_from_given_point(img, centre_x, centre_y)

    theta = find_angle_from_given_point(img, centre_x, centre_y)

    if inside_region:
        unselected_area = np.where((r < radius1) | (r > radius2) | (theta < theta1) | (theta > theta2))
        selected_area = np.where((r >= radius1) & (r <= radius2) & (theta >= theta1) & (theta <= theta2))
    else:
        unselected_area = np.where((r >= radius1) & (r <= radius2) & (theta >= theta1) & (theta <= theta2))
        selected_area = np.where((r < radius1) | (r > radius2) | (theta < theta1) | (theta > theta2))

    test[unselected_area] = 0

    return test, test[selected_area].flatten().tolist()


def crop_radial_arc_two_centres(img, centre_x1=None, centre_y1=None, centre_x2=None, centre_y2=None,
                                radius1=None, radius2=None, theta1=0, theta2=360, inside_region=True):
    test = np.copy(img)

    if centre_x1 is None:
        centre_x1 = img.shape[0] / 2
    if centre_y1 is None:
        centre_y1 = img.shape[1] / 2
    if centre_x2 is None:
        centre_x2 = img.shape[0] / 2
    if centre_y2 is None:
        centre_y2 = img.shape[1] / 2

    if radius1 is None:
        radius1 = 0

    if radius2 is None:
        if img.shape[1] > img.shape[0]:
            radius2 = img.shape[0] / 2
        else:
            radius2 = img.shape[1] / 2

    r1 = find_radial_distance_from_given_point(img, centre_x1, centre_y1)
    r2 = find_radial_distance_from_given_point(img, centre_x2, centre_y2)

    theta = find_angle_from_given_point(img, centre_x1, centre_y1)

    if inside_region:
        unselected_area = np.where((r1 < radius1) | (r2 > radius2) | (theta < theta1) | (theta > theta2))
        selected_area = np.where((r1 >= radius1) & (r2 <= radius2) & (theta >= theta1) & (theta <= theta2))
    else:
        unselected_area = np.where((r1 >= radius1) & (r1 <= radius2) & (theta >= theta1) & (theta <= theta2))
        selected_area = np.where((r2 < radius1) | (r2 > radius2) | (theta < theta1) | (theta > theta2))

    test[unselected_area] = 0

    return test, test[selected_area].flatten().tolist()
