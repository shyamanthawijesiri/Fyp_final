import cv2
import numpy as np

def classificationByDensity(resized):
    # SIZE = 512
    # resized = cv2.resize(resized, (SIZE, SIZE))
    # cv2.imshow("res233", resized)

    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    img_blur =cv2.GaussianBlur(img_gray, (3, 3), 0)
    (_, thresh) = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    l_b = np.array([32, 16, 24])
    u_b = np.array([255, 255, 255])

    # Initial mask
    mask = cv2.inRange(hsv, l_b, u_b)

    cnt, hierarchy = cv2.findContours(
        mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)

    # Marking the found contours in the img_copy
    # Sort the contours
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

    my_img_3 = np.zeros((512, 512, 1), dtype="uint8")

    cv2.drawContours(my_img_3, [cnt[0]], contourIdx=-1,
                     color=(255, 0, 0), thickness=2)

    cv2.fillPoly(my_img_3, pts=[cnt[0]], color=(255, 255, 255))

    result = cv2.bitwise_and(resized, resized, mask=my_img_3)
    # cv2.imshow('my_img_3',my_img_3)
    # cv2.imshow("res2", result)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
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

    my_img_3 = np.zeros((512, 512, 3), dtype="uint8")
    cv2.drawContours(my_img_3, contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=1)
    # print("No of total initial contours :", len(contours))

    centerPointsX = []
    centerPointsY = []
    centerCoordinates = []
    newContours = []
    contourArea = []

    remaining_contours = []

    contours.pop(0)
    remaining_contours.append(contours[0])
    my_img__ = np.zeros((512, 512, 3), dtype="uint8")
    cv2.drawContours(my_img__, remaining_contours, contourIdx=0,
                     color=(255, 0, 0), thickness=1)

    # print("Area of second largest contour: ", cv2.contourArea(remaining_contours[0]))

    #######################################################################
    for index, i in enumerate(contours):
        # print(i)
        if (cv2.contourArea(i) > 0.0 and cv2.contourArea(i) <= 1000.0):
            newContours.append(contours[index])
            contourArea.append(cv2.contourArea(i))
            c_ = contours[index]
            M_ = cv2.moments(c_)
            cx_ = int(M_['m10'] / M_['m00'])
            cy_ = int(M_['m01'] / M_['m00'])
            centerPointsX.append(cx_)
            centerPointsY.append(cy_)
            centerCoordinates.append((cx_, cy_))

    # print("Length of CX: ", len(centerPointsX))
    # print("Length of Cy: ", len(centerPointsY))
    # print("Length: ", len(centerCoordinates))
    # print("Average contour area: ", sum(contourArea)/len(centerPointsX))
    # print("Second largest contour area: ", contourArea[0])

    # my_img_ = np.zeros((512, 512, 3), dtype="uint8")
    #
    # cv2.drawContours(my_img_, contours, contourIdx=1,
    #                  color=(255, 0, 0), thickness=1)
    # cv2.imshow("Second largest contour", my_img__)

    c_0 = cnt[0]
    M = cv2.moments(c_0)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    t_m = tuple(c_0[c_0[:, :, 1].argmin()][0])
    b_m = tuple(c_0[c_0[:, :, 1].argmax()][0])
    r_m = tuple(c_0[c_0[:, :, 0].argmax()][0])
    l_m = tuple(c_0[c_0[:, :, 0].argmin()][0])

    mask=np.zeros((512,512),dtype='uint8')
    # divide contours
    cntours = cnt[0]
    hull = cv2.convexHull(cntours)
    res = cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
    cv2.line(res, (cx, 0), (cx, res.shape[0]), 0, 2)
    cv2.line(res, (0, cy), (res.shape[1], cy), 0, 2)
    conturs,_ = cv2.findContours(res,mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('res',res)
    # contours, _ = cv2.findContours(image=res, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resized,conturs,-1,(0,0,255),2)
    density=[]
    colors = [(0,255,0),(0,255,255),(0,0,255),(255,0,0)]
    for i in range(4):
        c = conturs[i]
        area = cv2.contourArea(c)
        k=0
        for crd in centerCoordinates:
            dist1 = cv2.pointPolygonTest(c,crd, True)
            if dist1>=0:
                k+=1
                cv2.circle(resized,crd,2,colors[i],-1)
        density.append(k/area)
    # print("Densities: ", density)
    # cv2.imshow('img2', resized)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    max_den= max(density)

    if(max_den>=0.004):
        # print("BetelRust Disease")
        return 2
    else:
        return 0
        # print("Other")

########################################################################################################################
#
# img= cv2.imread('Diseases/kanamadiri/13.jpg')
#
# classificationByDensity(img)