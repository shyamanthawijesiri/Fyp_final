import cv2
import numpy as np


def getImage(path):
    SIZE = 512
    img = cv2.imread(path)
    resized = cv2.resize(img, (SIZE, SIZE))
    return resized


def binaryImage(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.cv2.GaussianBlur(img_gray, (3, 3), 0)
    (_, thresh) = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def backgroundFreeLeaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
    return contours


def AreaCalculation(img):
    contours=getCountours(img)
    remaining_contours = []

    contours.pop(0)
    remaining_contours.append(contours[0])
    my_img__ = np.zeros((512, 512, 3), dtype="uint8")
    cv2.drawContours(my_img__, remaining_contours, contourIdx=0,
                     color=(255, 0, 0), thickness=1)
    secondLargestArea = cv2.contourArea((remaining_contours[0]))
    # print("Area of second largest contour: ", cv2.contourArea(remaining_contours[0]))
    return secondLargestArea

def classification(img):
    area = AreaCalculation(img)
    return area
    # if area <= 200:
    #     # print("Malakada")
    #     return 2
    # elif area >= 1000:
    #     # print("Bacterial")
    #     return 1
    # else:
    #     # print("Kanamediri haniya")
    #     return 3


########################################################################################
#
# img = getImage('Diseases/bacterial/IMG_9363.JPG')
# img = getImage('Diseases/malakada/2.jpg')






# a=AreaCalculation(img)
