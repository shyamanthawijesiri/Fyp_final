import cv2
import numpy as np
import math
# from matplotlib import pyplot as plt




def getImage(path,height=512,width=512):
    img = cv2.imread(path)
    resized = cv2.resize(img, (height, width))
    return resized


def binaryImage(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.cv2.GaussianBlur(img_gray, (3, 3), 0)
    (_, thresh) = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh


def getLeaf(binaryImage,orgImage):
    contours, _ = cv2.findContours(image=binaryImage, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)
    c,l_m,r_m,t_m,b_m = contourPoints(cnt[-2])
    contourImage = orgImage.copy()

    cv2.drawContours(contourImage, [cnt[-2]], -1, color=(255, 0, 0), thickness=2)
    mask = np.zeros_like(binaryImage)
    mask = cv2.fillPoly(mask, [cnt[-2]], 255, 1)
    leaf = cv2.bitwise_and(orgImage, orgImage, mask=mask)
    return contourImage, mask, leaf, cnt[-2]

def blackToWhite(img):
    white_pix = [255, 255, 255]
    black_pix = [0, 0, 0]
    white_img = img.copy()
    for y in range(0, white_img.shape[0]):
        for x in range(0,  white_img.shape[1]):
            if all(white_img[y,x] == black_pix):
                    white_img[y,x] = white_pix

    return  white_img
def translation(img,bImg,cnt):
    M = cv2.moments(cnt)
    cx =int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    M = np.float32([[1, 0, 250-cx], [0, 1, 250-cy]])
    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    shift_bImg = cv2.warpAffine(bImg, M, (bImg.shape[1], bImg.shape[0]))
    return shift_img,shift_bImg

def contourPoints(cnt):
    M = cv2.moments(cnt)
    c = (int(M['m10'] / M['m00']),int(M['m01'] / M['m00']))
    l_m = tuple(cnt[cnt[:, :, 0].argmin()][0])
    r_m = tuple(cnt[cnt[:, :, 0].argmax()][0])
    t_m = tuple(cnt[cnt[:, :, 1].argmin()][0])
    b_m = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return c,l_m,r_m,t_m,b_m


def extractVein(img):
    new_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    fil_2d = cv2.filter2D(img, -1, (5,5))
    blur = cv2.blur(img,(5,5))
    median_blur=cv2.medianBlur(img,5)
    med_val = np.median(img)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    sobelxy = cv2.Canny(img_blur,lower,upper)
    con, hierarchy = cv2.findContours(sobelxy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(con):
        area = cv2.contourArea(c)
        if area > 100:

             cv2.drawContours(new_img, con, i, color=(255, 0, 0), thickness=2)


    return new_img

def rotate(img,mask,cx,cy,b_m):
    w= cy - b_m[1]
    h = cx - b_m[0]
    cv2.circle(img,(b_m[0],b_m[1]),3,(0,0,255),-1)
    cv2.circle(img,(cx,cy),3,(0,0,255),-1)
    angle = math.atan(h/w)*180 / math.pi
    # print(angle)
    # print(cy,b_m[1],cx,b_m[0])
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    rot_mask = cv2.warpAffine(mask, M, (w, h))

    contours, _ = cv2.findContours(rot_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return rotated,rot_mask,contours


