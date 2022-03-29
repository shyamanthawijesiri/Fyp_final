import cv2
import numpy as np
from PIL import Image
import preprocessing as pre


def holes(orgLeaf):
    leaf =pre.blackToWhite(orgLeaf)
    leaf = cv2.cvtColor(leaf, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.bilateralFilter(leaf, d=7, sigmaSpace=75, sigmaColor=75)
    ret, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, )
    # Draw the contour
    img_hole = orgLeaf.copy()
    hole_color = []
    for i in range(len(contours) - 1):
        area = cv2.contourArea(contours[i])
        if area > 10:
            cv2.drawContours(img_hole, [contours[i]], contourIdx=-1, color=(255, 0, 0), thickness=1)
            c_0 = contours[i]
            M = cv2.moments(c_0)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            a = leaf[cy, cx]
            hole_color.append(a)

    return hole_color,img_hole
