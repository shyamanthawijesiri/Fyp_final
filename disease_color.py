import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt



def getImage(path,height=512,width=512):
    img = cv2.imread(path)
    resized = cv2.resize(img, (height, width))
    return resized


def getLeaf(img):
    SIZE = 512
    img = cv2.resize(img, (SIZE, SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_b = np.array([28, 31, 0])
    u_b = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, l_b, u_b)
    imgN = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    res = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    contours.pop(-1)
    d = []
    d.append(contours[-1])
    mask2 = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    cv2.fillPoly(mask2, pts=d, color=(255, 255, 255))
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    return mask2,res2


def getMeanStdDev(img,mask):
    b=[]
    g=[]
    r=[]
    # print(img[2][2][2])
    for x in range(512):
        for y in range(512):
            b.append(img[x][y][0])
            g.append(img[x][y][1])
            r.append(img[x][y][2])

    bm = np.amax(b)
    gm=np.amax(g)
    rm=np.amax(r)

    (mean, std) = cv2.meanStdDev(img)

    multiplier = float(mask.size) / cv2.countNonZero(mask)
    mean_N = tuple(multiplier * x for x in mean)
    std_N = tuple([multiplier * x for x in std])
    r_mean = mean_N[2][0]
    g_mean = mean_N[1][0]
    b_mean = mean_N[0][0]

    r_std = std_N[2][0]
    g_std = std_N[1][0]
    b_std = std_N[0][0]
    return (r_mean,g_mean,b_mean) ,(r_std,g_std,b_std),(bm,gm,rm)


def colorClassificationBstd(img):
    # SIZE = 512
    # resized = cv2.resize(img, (SIZE, SIZE))
    mask, leaf = getLeaf(img)
    mean, std, maxColor = getMeanStdDev(leaf, mask)
    # print(std[2])
    return std[2]
    # if std[2]<=900:
    #     # print("Bacterial")
    #     return 1
    # elif std[2]>=1600:
    #     # print("Malakada")
    #     return 2
    # else:
    #     # print("Kanamadiri")
    #     return 3

def colorClassificationGstd(img):
    SIZE = 512
    resized = cv2.resize(img, (SIZE, SIZE))
    mask, leaf = getLeaf(resized)
    mean, std, maxColor =getMeanStdDev(leaf, mask)
    print(std[1])
    if std[1]<=1700:
        print("Bacterial")
    elif std[1]>=3000:
        print("Malakada")
    else:
        print("Kanamadiri")


def colorClassificationRstd(img):
    SIZE = 512
    resized = cv2.resize(img, (SIZE, SIZE))
    mask, leaf = getLeaf(resized)
    mean, std, maxColor = getMeanStdDev(leaf, mask)
    print(std[0])
    if std[0]<=1700:
        print("Bacterial")
    elif std[0]>=3900:
        print("Malakada")
    else:
        print("Kanamadiri")


# img = cv2.imread('Diseases3/kanamadiri/20210207_121515.jpg')
# colorClassificationBstd(img)

