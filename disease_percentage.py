import cv2
import numpy as np
import preprocessing as pre

def percentage(frame):
    frame = pre.getImage(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_b = np.array([32, 16, 24])
    u_b = np.array([255, 255, 255])

    # Initial mask
    mask = cv2.inRange(hsv, l_b, u_b)

    contours, hierarchy = cv2.findContours(
        mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)
    contoursz_ = contours
    # Marking the found contours in the img_copy
    # Sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    ##################################################### Start of disease area cal

    #Disease Percentage
    contourArea=[]
    contoursz_ = sorted(contoursz_, key=cv2.contourArea, reverse=True)
    area_h = cv2.contourArea(contoursz_[0])
    contoursz_.pop(0)
    remaining_contours = []
    remaining_contours.append(contours[0])
    for index, i in enumerate(contoursz_):
        contourArea.append(cv2.contourArea(i))

    print("List of area values except largest area:", contourArea)
    print("Area h:", area_h )

    areaSum = sum(contourArea)

    print("Area l:", areaSum)
    DiseasePercentage= areaSum*100/area_h

    print("Disease Percentage:", DiseasePercentage)
    return DiseasePercentage