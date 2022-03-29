import cv2
from matplotlib import pyplot as plt
import preprocessing as gt
import numpy as np

# img = gt.getImage('images/kalu/kalu5.jpg')
# bImg = gt.binaryImage(img)
# cImg,mask, leaf, cnt = gt.getLeaf(bImg, img)
# print(mask.size)
def getMean(img):
    channels = cv2.mean(img)
    observation = np.array([(channels[2], channels[1], channels[0])])
    return observation

def getMeanStdDev(img,mask):
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

    return (r_mean,g_mean,b_mean),(r_std,g_std,b_std)

# print('ob',ob)

# return observation
# print(observation)
# s = plt.figure()
# s.add_subplot(2,1,1)
# plt.imshow(observation)
# plt.axis("off")
# plt.title('sudu')
# plt.show()
# s.add_subplot(2,1,2)
# # plt.imshow(observation2)
# plt.axis("off")
# plt.title('kalu')
# # plt.hist(leaf.ravel(),256,[1,70])
# plt.show()
