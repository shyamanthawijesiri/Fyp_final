import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import preprocessing as gt



def cntCenterBottom(cnt):
    c_0 = cnt
    M = cv2.moments(c_0)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    b_m = tuple(c_0[c_0[:, :, 1].argmax()][0])
    return cx,cy,b_m

def petiolePoint(img,cx,cy,b_m,lap):
    gray = img
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.005)
    dst = cv2.dilate(dst, (5,5))
    img2 = cv2.cvtColor(lap,cv2.COLOR_GRAY2BGR)
    crd = []
    xN=yN=0
    for y in range(cy-150,cy):
        for x in range(cx-40,cx+40):
            if dst[y,x] >  0.075 * dst.max() :
                img2[y,x] = [0,255,0]
                crd.append((y,x))
    D = math.sqrt((cy)**2 + (cx)**2)
    for i in crd:
        Dnew = math.sqrt((i[0] - cy)**2 + (i[1] - cx)**2)
        if Dnew < D:
            xN = i[1]
            yN  = i[0]
            D = Dnew
    img2[yN, xN] = [0, 0, 255]
    img2[10, 50] = [0, 0, 255]
    cv2.line(img2,(xN,yN),(b_m[0],b_m[1]),(0,0,255), 1)
    # print(xN,yN)
    return img2, xN, yN

def laplacian(img):
    gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gr = cv2.bilateralFilter(gr, 3, 50, 50)
    lap = cv2.Laplacian(gr, cv2.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
    n = plt.hist(lap.ravel(), 256, [1, 255])
    numpy_arr = np.array(n[0])
    global index
    # print(sorted(n[0]))
    for i,c in enumerate(sorted(n[0],reverse=True)):
        if c<600 :
            index = np.where(numpy_arr==c)
            break
    # print(index)
    thresh_val = n[1][index][0]
    # print(thresh_val)
    ret, thresh = cv2.threshold(lap, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh

def removePetiole(mask,img):
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, stalk = cv2.threshold(dist_transform, 0.099 * dist_transform.max(), 255, 0)
    stalk = np.uint8(stalk)
    cnt,_ = cv2.findContours(stalk, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=cv2.contourArea,reverse=True)
    return cnt[0]
def calculateGap(cnt,img,xn):
    right = []
    left = []
    for c in cnt:
        x = c[0][0]
        # print(c[0])
        # print(x)
        if x > xn:
            right.append(c)
        else:
            left.append(c)
    con_right = np.array(right)
    con_left = np.array(left)
    t_r = tuple(con_right[con_right[:, :, 1].argmin()][0])
    t_l = tuple(con_left[con_left[:, :, 1].argmin()][0])
    cv2.drawContours(img,[con_left],-1,(0,255,0),2)
    cv2.drawContours(img,[con_right],-1,(0,255,255),2)
    cv2.circle(img,(t_r[0],t_r[1]),5,(0,255,0),-1)
    cv2.line(img,(t_r[0],t_r[1]),(t_l[0],t_r[1]),(0,255,0),2)
    cv2.circle(img,(t_l[0],t_l[1]),5,(0,0,255),-1)
    cv2.line(img,(t_l[0],t_l[1]),(t_r[0],t_l[1]),(0,0,255),2)
    d = abs(t_r[1] - t_l[1])
    cv2.putText(img, 'Height difference = {}units'.format(d), (50, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2,
                cv2.LINE_AA)
    cv2.circle(img, (t_l[0], t_l[1]), 5, (0, 0, 255), -1) 
    # cv2.imshow('imgcon',img)
    return img, d
def getGap(img,cnt,leaf,mask):
    c, l_m, r_m, t_m, b_m = gt.contourPoints(cnt)
    rotated, rotated_mask, rotated_cnt = gt.rotate(leaf, mask, c[0], c[1], b_m)
    remove_cnt = removePetiole(rotated_mask, img)
    # cx, cy, b_m = cntCenterBottom(cnt)
    lap = laplacian(rotated)
    pet,xn,yn = petiolePoint(rotated_mask,c[0],c[1],b_m,lap)
    final_img, gap = calculateGap(cnt,img,xn)
    # cv2.imshow('pet-1', pet)
    # cv2.drawContours(img,[remove_cnt],-1,(0,255,255),2)
    # cv2.imshow('img',img)
    return final_img, gap
