import numpy as np
import cv2
import time


def showimage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mean(a):
    return sum(a) / len(a)


def gradeintimage(img):
    positions1 = []
    # positions2 = []
    image = cv2.imread(img)

    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a = time.time()
    for i in range(len(imagegray)):
        for j in range(len(imagegray[i]) - 2):
            if imagegray[i][j] == 128 and imagegray[i + 1][j] < 10:
                if imagegray[i][j - 1] < 10 and imagegray[i][j - 2] < 10 and imagegray[i][j - 3] < 10 and imagegray[i][
                    j - 4] < 10 and imagegray[i][j - 5] > 10:
                    if imagegray[i + 1][j] < 10 and imagegray[i - 1][j] > 10 and imagegray[i - 2][j] > 10 and \
                            imagegray[i - 3][j] > 10:
                        image[i][j] = [255, 255, 0]
                        image[i][j - 1] = [255, 255, 0]
                        image[i][j - 2] = [255, 255, 0]
                        image[i][j - 3] = [255, 255, 0]
                        image[i][j - 4] = [255, 255, 0]
                        print(i, j)
    b = time.time()
    showimage("", image)
    c = b - a
    print(c)


def numpygradient(imagename):
    image = cv2.imread(imagename)
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p = np.where(imagegray == 128)
    a = p[0]
    b = p[1]
    atime = time.time()
    for i in range(len(p[0])):
        if imagegray[a[i] + 1][b[i]] < 10 and imagegray[a[i]][b[i] - 1] < 10 and imagegray[a[i]][b[i] - 2] < 10 and \
                imagegray[a[i]][b[i] - 3] < 10 and imagegray[a[i]][b[i] - 4] < 10 and imagegray[a[i]][b[i] - 5] > 10:
            if imagegray[a[i] - 1][b[i]] > 10 and imagegray[a[i] - 2][b[i]] > 10 and imagegray[a[i] - 3][b[i]] > 10:
                image[a[i]][b[i]] = [255, 255, 0]
                image[a[i]][b[i] - 1] = [255, 255, 0]
                image[a[i]][b[i] - 2] = [255, 255, 0]
                image[a[i]][b[i] - 3] = [255, 255, 0]
                image[a[i]][b[i] - 4] = [255, 255, 0]
    btime = time.time()
    c = btime - atime
    print(c)
    showimage("", image)


imagename = "side_tooth_img1.jpg"

image = cv2.imread(imagename)
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showimage("",imagegray)

# showimage("", image)
# gradeintimage(imagename)
# numpygradient(imagename)
