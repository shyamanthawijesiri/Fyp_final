import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

SIZE=512
def getImage(path):
    SIZE = 512
    img = cv2.imread(path)
    resized = cv2.resize(img, (SIZE, SIZE))
    return resized

def backgroundFreeLeaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # l_b = np.array([28,31,0])
    l_b = np.array([32, 16, 24])
    u_b = np.array([255, 255, 255])

    # Initial mask
    mask = cv2.inRange(hsv, l_b, u_b)
    contours, hierarchy = cv2.findContours(
        mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)

    # Marking the found contours in the img_copy
    # Sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    my_img_3 = np.zeros((512, 512, 1), dtype="uint8")
    cv2.drawContours(my_img_3, [contours[0]], contourIdx=-1, color=(255, 0, 0), thickness=2)
    cv2.fillPoly(my_img_3, pts=[contours[0]], color=(255, 255, 255))
    res = cv2.bitwise_and(frame, frame, mask=my_img_3)
    return res

def getCountours(img):
    img = backgroundFreeLeaf(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # l_b = np.array([28,31,0])
    l_b = np.array([32, 16, 24])
    u_b = np.array([255, 255, 255])

    # Initial mask
    mask = cv2.inRange(hsv, l_b, u_b)

    contours, hierarchy = cv2.findContours(
        mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)

    # Marking the found contours in the img_copy
    # Sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # print("Number of contours = " + str(len(contours)))
    contours.pop(0)
    return contours


def getMax(a):
    # print(len(a))
    max = 0
    for z in range(len(a) - 1):
        gap = abs(int(a[z + 1] )- int(a[z]))
        if (max < gap):
            point1=z
            point2=z+1
            max = gap
    # print(point1)
    # print(point2)
    # print("**")
    v1=a[point1]
    v2=a[point2]
    change=int(v2)-int(v1)
    return change,point2



def colorVariation(img):
    # img = cv2.resize(img, (SIZE, SIZE))
    img=backgroundFreeLeaf(img)
    contours=getCountours(img)
    d=[]
    colorValueB =[]
    colorValueG = []
    colorValueR = []
    z=[]

    area = cv2.contourArea(contours[0])
    if area<=200:
        size=15
    elif area>=1000:
        size=45
    else:
        size=20

    # The first order of the contours
    c_0 = contours[0]
    # image moment
    M = cv2.moments(c_0)
    # The centroid point
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # print(cx, cy)
    x=cx
    for x in range(x,x+size):
            b = img[cy,x][0]
            g = img[cy,x][1]
            r = img[cy,x][2]

            colorValueB.append(b)
            colorValueG.append(g)
            colorValueR.append(r)
            z.append(x)

    d.append(contours[0])
    for cx in range(cx,cx+size):
            img[cy, cx] = (255, 255, 255)

    # print(colorValueG)

    # print("Max gap of G Channel =", getMax(colorValueG))
    # classfication(getMax(colorValueG))
    # # print(colorValueR)
    # print("Max gap of R Channel =", getMax(colorValueR))
    maxG,pointG=getMax(colorValueG)
    maxR,pointR=getMax(colorValueR)
    return maxG,pointG,maxR,pointR
    # classfication2(maxG,pointG,maxR,pointR)
    # print("****")

def classfication2(img):
    maxG,pointG,maxR,pointR=colorVariation(img)
    # print(maxG)

    if abs(maxG )>= 20 and abs(maxR) >= 20 :
        if (maxG > 0 and maxR > 0) and (pointG>=10 or pointR>=10) :
            # print("Bacterial")
            return 1
        else:
            # print("Other")
            return 0
    else:
        # print("Other")
        return 0







