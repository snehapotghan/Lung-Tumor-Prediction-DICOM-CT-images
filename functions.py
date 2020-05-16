import cv2
import numpy as np
import os
import dicom
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
slices=[]
def dicomRead(path):

    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                slices.append(dicom.read_file(os.path.join(dirName, filename)))
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    image = np.stack([s.pixel_array for s in slices])
    return image[0:255,:,:]
def floodfill(image, start_point, value):
    height, width = image.shape[:2]
    points = [start_point]
    flag = [[0 for j in range(width)] for i in range(height)]
    flag[start_point[0]][start_point[1]] = 1
    origin_value = image[start_point[0]][start_point[1]]
    while len(points) > 0:
        pt = points.pop(0)
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        for x, y in zip(dx, dy):
            if (0 <= pt[0] + x < height and 0 <= pt[1] + y < width and
                        origin_value == image[pt[0] + x][pt[1] + y] and
                        flag[pt[0] + x][pt[1] + y] == 0):
                flag[pt[0] + x][pt[1] + y] = 1
                points.append((pt[0] + x, pt[1] + y))
        image[pt[0]][pt[1]] = value
    return image


def switch_pixels(image, origin_value, value):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == origin_value:
                image[i][j] = value
    return image

def morphology_open(image):
    # morphology open operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def segment(image):
    threashold=625
    mask = np.copy(image)

    mask = morphology_open(mask)
    ret, mask = cv2.threshold(mask, threashold, 255, cv2.THRESH_BINARY_INV)

    # set margin to black
    h, w = mask.shape[:2]
    for i in range(h):
        for j in range(w):
            if ((i == 0 or j == 0 or i == h - 1 or j == w - 1) and
                        mask[i][j] != 0):
                mask = floodfill(mask, (i, j), 0)

    # fill holes in middle
    mask = floodfill(mask, (0, 0), -1)
    mask = switch_pixels(mask, 0, 255)
    mask = switch_pixels(mask, -1, 0)
    return mask


def test3D(ConstPixelDims = None):
    xx, yy = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512))
    X = xx
    Y = yy
    #Z = i
    ax2 = gca(projection='3d')
    off = 4000 / (ConstPixelDims[2] - 1)
    for i in range(ConstPixelDims[2]):  # ConstPixelDims[2]
        tempImage = ArrayDicom[:, :, i]
        tempImage = np.ma.masked_where(tempImage < 1200, tempImage)
        print("i : " + str(i))
        print("tempImage.shape[0] " + str(tempImage.shape[0]))
        print("tempImage.shape[1] " + str(tempImage.shape[1]))
        x, y = ogrid[0:tempImage.shape[0], 0:tempImage.shape[1]]
        ##print("x : " + str(x) + " :: y : " + str(y))
        Z = 10 * np.ones(X.shape)
        ax = gca(projection='3d')

        # ax.plot_surface(x, y, Z, rstride=5, cstride=5, facecolors=tempImage,cmap='gray' )

        ax2.contourf(X, Y, tempImage, zdir='z', offset=i * off, antialiased=True)

        # ax2.set_zlim((0.,1.))

    show()