import pandas as pd
# import matplotlib.pyplot as plt
import os
import csv

import cv2
import preprocessing as gt
import width_length as dimension
import color_detect as color
import height_difference as gap


def width():
    W_KORI_MIN = 0
    W_KORI_MAX = 12

    W_KALU_MIN = 17

def widthKorikan(width):
    W_MAX = 20
    W_KORI_MIN = 0
    W_KORI_MAX = 12
    global p_kori

    if width > W_KORI_MIN and width <= W_KORI_MAX:
        p_kori = 1
    elif width > W_KORI_MAX:
        p_kori = (W_MAX - width) / (W_MAX - W_KORI_MAX)

    return p_kori


def lengthKorikan(length):
    L_MAX = 30
    L_KORI_MIN = 0
    L_KORI_MAX = 20
    global p_kori

    if length > L_KORI_MIN and length<= L_KORI_MAX:
        p_kori = 1

    elif length > L_KORI_MAX:
        p_kori = (L_MAX - length)/(L_MAX-L_KORI_MAX)

    return p_kori


def widthKalu(width):
    W_KALU_MIN = 17
    # W_KALU_MAX = 20
    global p_kalu

    if width >= W_KALU_MIN:
        p_kalu = 1
    elif width < W_KALU_MIN:
        p_kalu = (width) / (W_KALU_MIN)

    return p_kalu

def lengthKalu(length):
    L_KALU_MIN = 26
    # W_KALU_MAX = 20
    global p_kalu

    if length >= L_KALU_MIN:
        p_kalu = 1
    elif length < L_KALU_MIN:
        p_kalu = length / (L_KALU_MIN)

    return p_kalu

def heightDefference(gap):
    MIN_GAP = 8
    global gap_kanda
    global gap_piduna
    if gap<=MIN_GAP:
        gap_kanda =1
        gap_piduna = gap/MIN_GAP
    else:
        gap_piduna = 1
        gap_kanda = (30-gap)/30
    return gap_kanda,gap_piduna

def bMeanRan(bMean):
    MAX = 60
    BMEAN_RAN_MIN =0
    BMEAN_RAN_MAX = 29
    global p_ran
    if bMean > BMEAN_RAN_MIN and bMean<= BMEAN_RAN_MAX:
        p_ran = 1
    elif bMean>BMEAN_RAN_MAX:
        p_ran = (MAX - bMean)/(MAX-BMEAN_RAN_MAX)
    return p_ran

def bMeanKanda(bMean):
    BMEAN_KANDA_MIN =29
    # RMEAN_RAN_MAX =40
    global p_kanda
    if bMean > BMEAN_KANDA_MIN:
        p_kanda = 1
    elif bMean<BMEAN_KANDA_MIN:
        p_kanda = (bMean)/(BMEAN_KANDA_MIN)
    return p_kanda

def gMeanKalu(gMean):
    MAX = 90
    GMEAN_KALU_MIN =0
    GMEAN_KALU_MAX =60
    global p_kalu

    if gMean > GMEAN_KALU_MIN and gMean <= GMEAN_KALU_MAX:
        p_kalu = 1
    elif gMean>GMEAN_KALU_MAX:
        p_kalu = (MAX - gMean)/(MAX-GMEAN_KALU_MAX)
    return p_kalu

def gMeanSudu(gMean):
    GMEAN_SUDU_MIN =61
    # RMEAN_RAN_MAX =40
    global p_sudu
    if gMean > GMEAN_SUDU_MIN:
        p_sudu = 1
    elif gMean < GMEAN_SUDU_MIN:
        p_sudu = gMean/GMEAN_SUDU_MIN
    return p_sudu

def getColor(rMean,gStd):
    RMEAN_RAN_MAX = 25
    RMEAN_KALU_SUDU_MAX = 44
    RMEAN_KANDA_MAX = 80
    GSTD_KALU_MAX = 125
    GSTD_SUDU_MAX = 220
    global p_sudu
    global p_kalu
    global p_kanda
    global p_ran
    if rMean<RMEAN_RAN_MAX:
        p_ran = 1
        p_kalu = p_sudu = rMean/RMEAN_RAN_MAX
        p_kanda = rMean/RMEAN_KALU_SUDU_MAX
    elif rMean<RMEAN_KALU_SUDU_MAX:
        p_ran =(RMEAN_KALU_SUDU_MAX -rMean) /( RMEAN_KALU_SUDU_MAX-RMEAN_RAN_MAX)
        p_kanda = (rMean-RMEAN_RAN_MAX)/( RMEAN_KALU_SUDU_MAX-RMEAN_RAN_MAX)
        if gStd< GSTD_KALU_MAX:
            p_kalu = 1
            p_sudu = gStd/ GSTD_KALU_MAX
        else:
            p_sudu =1
            p_kalu = (GSTD_SUDU_MAX-gStd)/(GSTD_SUDU_MAX - GSTD_KALU_MAX)
    else:
        p_kanda = 1
        p_kalu = p_sudu = (RMEAN_KANDA_MAX-rMean)/(RMEAN_KANDA_MAX-RMEAN_KALU_SUDU_MAX)
        p_ran =(RMEAN_KANDA_MAX-rMean)/(RMEAN_KANDA_MAX-RMEAN_RAN_MAX)
    return p_kalu,p_sudu,p_kanda,p_ran

def featureExtraction(path):
    img = gt.getImage(path)
    # cv2.imshow('img',img)
    bImg = gt.binaryImage(img)
    cImg,mask, leaf, cnt = gt.getLeaf(bImg, img)
    try:
        _, width, height = dimension.getDimession(img)
        mean, std = color.getMeanStdDev(leaf, mask)
        _, dif = gap.getGap(img, cnt, leaf, mask)
        return width, height,mean[0],mean[1],2,dif
    except Exception as ex:
        err = type(ex).__name__
        print(err)

def classification(width,length,rMean,gMean,gStd,dif):

    pkalu = widthKalu(width)*lengthKalu(length)
    pkori = widthKorikan(width)*lengthKorikan(length)
    scores = {}
    if pkori == 1:
        scores["category"] = "Korikan"
        scores["korikan"] = pkori
        return scores
    else:
        if dif >5 and dif < 10:
            pran = getColor(rMean, gStd)[3]
            pkanda = getColor(rMean, gStd)[2]
            pkalu = pkalu * getColor(rMean, gStd)[0]
            psudu = getColor(rMean, gStd)[1]
            scores["kalu"] = pkalu
            scores["Sudu"] = psudu
            scores["Kanda Kola"] = pkanda
            scores["Ran Kola"] = pran
            scores["korikan"] = pkori
            return scores

        elif dif < 5:
            pran = bMeanRan(rMean)
            pkanda = bMeanKanda(rMean)
            scores["kalu"] = pkalu
            scores["Kanda Kola"] = pkanda
            scores["Ran Kola"] = pran
            scores["korikan"] = pkori
            return scores
        else:
            pkalu = pkalu * gMeanKalu(gMean)
            psudu = gMeanSudu(gMean)
            scores["kalu"] = pkalu
            scores["Sudu"] = psudu
            scores["korikan"] = pkori
            return scores

    # gap_kanda, gap_piduna = heightDefference(dif)
    # # if pkalu == 1 or pkori == 1:
    # #     print("kalu prob = {} \nkorikan prob = {}".format(pkalu, pkori))
    # #     return
    # # else:
    # pran = rMeanRan(rMean)*gap_kanda
    # pkanda = rMeanKanda(rMean)*gap_kanda
    # pkalu = pkalu * gMeanKalu(gMean)*gap_piduna
    # psudu = gMeanSudu(gMean)*gap_piduna
    # max_prob = [pkalu,psudu,pkanda,pran,pkori]
    # index = max_prob.index(max(max_prob))
    # return result_label[index],pkalu,psudu,pkanda,pran,pkori
    # if index == 0:
    #     return 'kalu'
    # elif index == 1:
    #     return 'sudu'
    # elif index == 2:
    #     return 'kanda'
    # elif index == 3:
    #     return 'ran'
    # elif index == 4:
    #     return 'kori'
    # print("ran prob = {} \nkanda prob = {}".format(pran, pkanda))
    # print("kalu prob = {} \nsudu prob = {}".format(pkalu, psudu))
    # print("korikan prob = {}".format(pkori))
    # elif dif <=5:
    #     pran=rMeanRan(rMean)
    #     pkanda = rMeanKanda(rMean)
    #     print("ran prob = {} \nkanda prob = {}".format(pran, pkanda))
    # elif dif >5:
    #     pkalu = pkalu*gMeanKalu(gMean)
    #     psudu = gMeanSudu(gMean)
    #     print("kalu prob = {} \nsudu prob = {}".format(pkalu, psudu))


# width,length,rMean,gMean,dif = featureExtraction('images/kalu/sudu3.jpg')

# classification(width,length,rMean,gMean,dif)
# print(dif)
def predict():
    categories = ['korikan','kalu','sudu']
    datadir = 'C:/Users/shyamantha/Documents/final year project/images/catergory/dataset/validation'
    f = open('result.csv', 'w', newline='')
    fieldnames = ['Img_name', 'Category','Predict_result',]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i in categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        print(path)
        for img_name in os.listdir(path):
            try:
                width, length, rMean, gMean, dif = featureExtraction(os.path.join(path, img_name))
                print("Width => {}".format(width))
                print("Length => {}".format(length))
                print("Rmean => {}".format(rMean))
                print("Gmean => {}".format(gMean))
                print("Gap => {}".format(dif))
                catergory = classification(width, length, rMean, gMean, dif)
                writer.writerow({
                    fieldnames[0]: img_name, fieldnames[1]: i, fieldnames[2]: catergory
                })

            except Exception as ex:
                err = type(ex).__name__
                # writer.writerow({fieldnames[0]: img_name, fieldnames[1]: i, fieldnames[2]: err, fieldnames[3]: err})
                print(err)
                print(ex)


# cv2.waitKey(0)
# cv2.destroyAllWindows()