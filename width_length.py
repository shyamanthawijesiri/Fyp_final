import cv2
import numpy as np
import preprocessing as gt


#
# def binary(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_blur = cv2.bilateralFilter(img, d=7, sigmaSpace=75, sigmaColor=75)
#     img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
#     a = img_gray.max()
#     # a = img_blur.max()
#     _, thresh = cv2.threshold(img_gray, a / 2 + 60, a, cv2.THRESH_BINARY)
#     return thresh

def warpMask(frame):

    l_b = np.array([0,0, 255])
    u_b = np.array([0, 0, 255])
    mask = cv2.inRange(frame, l_b, u_b)
    return  mask

def reorder(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
def imgWarpF(img,points,w,h,pad=0):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp
def getContours(img,biImg,draw=False):
    i=-1
    contours, _ = cv2.findContours(biImg,mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt[i], True)
    approx = cv2.approxPolyDP(cnt[i],0.02*perimeter, True)
    if draw:
        cv2.drawContours(img,[cnt[i]],-1,(255,0,0),2)
    return cnt,approx

def findDis(pts1,pts2):
    # print(pts1,pts2)
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

def getDimession(img):

    biImg = gt.binaryImage(img)
    scale = 3
    w = 210 * scale
    h = 297 * scale
    org_img = img.copy()
    contours, _ = cv2.findContours(biImg,mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)
    c = cnt[-2]
    x, y, w1, h1 = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w1//2, y + h1//2), (0, 0, 255), -1)
    d = cnt[-1]
    perimeter = cv2.arcLength(d,True)
    approx = cv2.approxPolyDP(d,0.02* perimeter, True)
    points = reorder(approx)
    imgWarp = imgWarpF(img,points,w,h)
    bi2 = warpMask(imgWarp)
    con,approx = getContours(img,bi2)
    nPoints = reorder(approx)

    nW = round((findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)*2
    nH = round((findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)*2
    cv2.putText(org_img, 'width = {}cm'.format(nW), (50,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0,0), 2, cv2.LINE_AA)
    cv2.putText(org_img, 'length = {}cm'.format(nH), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0,0), 2, cv2.LINE_AA)
    # print(nW,nH)
    # cv2.imshow('org_img', org_img)
    return org_img,nW,nH



