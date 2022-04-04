import cv2
import numpy as np
import matplotlib.pyplot as plt


def getImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    return img


def binaryImage(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.cv2.GaussianBlur(img_gray, (3, 3), 0)
    (_, thresh) = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def getLeaf(binaryImage, orgImage):
    contours, _ = cv2.findContours(image=binaryImage, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)
    contourImage = orgImage.copy()
    cv2.drawContours(contourImage, [cnt[-2]], -1, color=(255, 0, 0), thickness=2)
    mask = np.zeros_like(binaryImage)
    mask = cv2.fillPoly(mask, [cnt[-2]], 255, 1)
    leaf = cv2.bitwise_and(orgImage, orgImage, mask=mask)

    return contourImage, mask, leaf, cnt[-2]


def cntCenter(cnt):
    c_0 = cnt
    M = cv2.moments(c_0)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    b_m = tuple(c_0[c_0[:, :, 1].argmax()][0])
    t_m = tuple(c_0[c_0[:, :, 1].argmin()][0])
    return cx, cy, b_m, t_m


def cannyImage(mask):
    canny = cv2.Canny(mask, 100, 100)
    return canny


def measureDistance(canny, cx, cy, b_m, t_m):
    h = canny.shape[0]
    w = canny.shape[1]
    bgr_canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    left_t = []
    left_b = []
    right_t = []
    right_b = []
    for y in range(cy, t_m[1] + 80, -1):
        for xl in range(cx, 0, -1):
            if canny[y, xl] == 255:
                #print(xl)
                left_t.append(cx-xl)
                cv2.circle(bgr_canny, (xl, y), 5, (0, 0, 255), -1)
                break
        for xr in range(cx, w):
            if canny[y, xr] == 255:
                #print(xr)
                right_t.append(xr-cx)
                cv2.circle(bgr_canny, (xr, y), 5, (0, 255, 255), -1)
                break
    for y in range(cy, b_m[1] - 40):
        for xl in range(cx, 0, -1):
            if canny[y, xl] == 255:
                # print(xl)
                left_b.append(cx-xl)
                cv2.circle(bgr_canny, (xl, y), 5, (255, 0, 0), -1)
                break
        for xr in range(cx, w):
            if canny[y, xr] == 255:
                # print(xr)
                right_b.append(xr-cx)
                cv2.circle(bgr_canny, (xr, y), 5, (0, 255, 0), -1)
                break

    # print(right_t)
    # print(left_t)
    # print(right_b)
    # print(left_b)
    return right_t, left_t, right_b, left_b



def drawGraph(r_t, l_t, r_b, l_b):
    r_top = np.array(r_t)
    #print(r_top)
    l_top = np.array(l_t)
    #print(l_top)
    r_bottom = np.array(r_b)
    l_bottom = np.array(l_b)
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title('right_top')
    axs[0, 0].plot(r_top, linestyle='dotted')
    axs[0, 1].set_title('left_top')
    axs[0, 1].plot(l_top, linestyle='dotted')
    axs[1, 0].set_title('right_bottom')
    axs[1, 0].plot(r_bottom, linestyle='dotted')
    fig.tight_layout(pad=1.0)
    axs[1, 1].set_title('left_bottom')
    axs[1, 1].plot(l_bottom, linestyle='dotted')
    plt.show()
def detectMutation(r_t, l_t, r_b, l_b):
    r_top = np.array(r_t)
    l_top = np.array(l_t)
    r_bottom = np.array(r_b)
    l_bottom = np.array(l_b)
    mution = 0
    for x in r_top:
        print(x)
        # gap = np.diff([l_top[0]] + l_top)
    gaprt = np.diff(r_top)
    print("Gap of right top =", gaprt)

    for y in range(gaprt.shape[0]):
        if abs(gaprt[y]) > 10:
            print("Mutation")
            break

    for x in l_top:
        print(x)
    gaplt = np.diff(l_top)
    print("Gap of left top =", gaplt)

    for y in range(gaplt.shape[0]):
        if abs(gaplt[y]) > 10:
            print("Mutation")
            break

    for x in r_bottom:
        print(x)
    gaprb = np.diff(r_bottom)
    print("Gap of right bottom =", gaprb)

    for y in range(gaprb.shape[0]):
        if abs(gaprb[y]) > 10:
            print("Mutation")
            break

    for x in l_bottom:
        print(x)
    gaplb = np.diff(l_bottom)
    print("Gap of left bottom=", gaplb)

    for y in range(gaplb.shape[0]):
        if abs(gaplb[y]) > abs(10):
            print("Mutation")
            break







img = getImage('BA4.jpg')
#img = getImage('KMH2.jpg')
#img = getImage('kananopatches.jpg')

bImg = binaryImage(img)
cImg, mask, leaf, cnt = getLeaf(bImg, img)

cx, cy, b_m, t_m = cntCenter(cnt)
can = cannyImage(mask)
r_t, l_t, r_b, l_b = measureDistance(can, cx, cy, b_m, t_m)
# drawGraph(r_t, l_t, r_b, l_b)
# print(cx,cy)


# cv2.imshow('mask',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
