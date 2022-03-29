import pandas as pd
import pickle

import os
import mahotas as mt
import cv2 as cv
import numpy as np
import csv
import re

import preprocessing as gt

# def readImages(path):
#     for dirpath,dirname,file in os.walk(path):
#         print(len(file))
#         file.sort();
#         file=file[1:]
#         return(file)

def extract_feature(image):
    (mean, std) = cv.meanStdDev(image)
    color_feature = np.array(mean)
    color_feature = np.concatenate([color_feature, std]).flatten()
    ##Texture Feature
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)
    ## Shape Features
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh.copy(), 1, 2)
    cnt = contours[0]
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    shape = np.array([])
    shape = np.append(shape, area)
    shape = np.append(shape, perimeter)
    ht_mean = np.concatenate([ht_mean, color_feature]).flatten()
    ht_mean = np.concatenate([ht_mean, shape]).flatten()
    return (ht_mean)

def create_csv(img):
    mydata = [
        ['energy', 'contrast', 'correlation', 'variance', 'inverse difference moment', 'sum average', 'sum variance',
         'sum entropy', 'entropy', 'difference variance', 'difference entropy', 'info_corr',
         'maximal_corr_coeff', 'mean_B', 'mean_G', 'mean_R', 'std_B', 'std_G', 'std_R', 'area', 'perimeter']]


    feature = extract_feature(img)
    feature = feature.tolist()
    mydata.append(feature)
    myfile = open('inputImage.csv', 'w')
    with myfile:
        writer = csv.writer(myfile)
        writer.writerows(mydata)

# img = cv.imread('train/kanamediri7.jpg')
# create_csv(img)



def healthyLeafClassification(leaf):
#     img = gt.getImage(img)
#     bImg = gt.binaryImage(img)
#     cImg, mask, leaf, cnt = gt.getLeaf(bImg, img)
#     cv.imshow('img',leaf)
    create_csv(leaf)
    test = pd.read_csv('inputImage.csv', sep=',')
    fileName = 'healthy_training_model'
    load_model = pickle.load(open(fileName, 'rb'))
    results = load_model.predict(test)
    # if (results == 0):
    #     print('healthy')
    # elif (results == 1):
    #     print('unhealthy')
    return results
# img = cv.imread('train/kanamediri7.jpg')
# classify('images/kalu/kalu12.jpg')
# classify('images/kanamediriya/kana1.jpg')

cv.waitKey(0)
cv.destroyAllWindows()




# for result in results:
#     if (result == 0):
#         print('healthy')
#     elif (result == 1):
#         print('bacterial')
#     elif (result == 2):
#         print('kanamediri')
#     elif (result == 3):
#         print('malakada')