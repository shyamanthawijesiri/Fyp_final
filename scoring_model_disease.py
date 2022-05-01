import color_variation as c_variation
import dot_density as d_density
import area_classfication as area
import disease_color as color

# 0 - others (kana,mala)
# 1 - bacterial
# 2 - malakada
# 3 - kanamediri

import preprocessing as gt


def colorVariation(img):
    disease = c_variation.classfication2(img)
    # print(disease)
    if disease == 0 :
        b= 0.5
        k = 0.25
        m = 0.25
        return b,k,m
    else:
        b = 0.2
        k = 0.4
        m = 0.4
        return b, k, m


# def diseaseColor(img):
#     c = color.colorClassificationBstd(img)
#     # print(c)
#     if c == 1:
#         b = 0.5
#         k = 0.25
#         m = 0.25
#         return b, k, m
#     elif c== 2:
#         b = 0.25
#         k = 0.25
#         m = 0.5
#         return b, k, m
#     elif c == 3:
#         b = 0.25
#         k = 0.5
#         m = 0.25
#         return b, k, m

def diseaseColor(img):
    c = color.colorClassificationBstd(img)
    K = 900
    M = 1600
    MAX = 14500
    # print(c)
    if c <=K:
        pB = 1
        pK = c / K
        pM =  c/M
        return pB, pK, pM
    elif c <= M:
        pB = (M-c)/(M-K)
        pK =  1
        pM = (c-K)/(M-K)
        return pB, pK, pM
    else:
        pB = (MAX - c) / (MAX - K)
        pK = (MAX - c) / (MAX - M)
        pM = 1
        return pB, pK, pM

# def diseaseArea(img):
#     a = area.classification(img)
#
#     # print(c)
#     if a == 1:
#         b = 0.5
#         k = 0.25
#         m = 0.25
#         return b, k, m
#     elif a == 2:
#         b = 0.25
#         k = 0.25
#         m = 0.5
#         return b, k, m
#     elif a == 3:
#         b = 0.25
#         k = 0.5
#         m = 0.25
#         return b, k, m

def diseaseArea(img):
    a = area.classification(img)
    K = 200
    B = 1000
    MAX = 6500

    if a <=K:
        pM = 1
        pK = a/K
        pB = a/B
        return pB,pK,pM
    elif a<=B:
        pM = (B-a)/(B-K)
        pK = 1
        pB = (a-K)/(B-K)
        return pB, pK, pM
    else:
        pM = (MAX-a) / (MAX - K)
        pK = (MAX-a)/(MAX-B)
        pB = 1
        return pB, pK, pM

def density(img):
    density = d_density.classificationByDensity(img)
    # print(density)
    if density == 2:
        b= 0.25
        k = 0.25
        m = 0.5
        return b,k,m
    else:
        b = 0.4
        k = 0.4
        m = 0.2
        return b, k, m

def classification(img_path):
    img = gt.getImage(img_path)
    result_label = ['Bacterial Leaf Blight', 'Kalamadiri Haniya (early stage)', 'Betel Rust']
    BcolorVariation, KcolorVariation, McolorVariation = colorVariation(img)
    BdiseaseColor, KdiseaseColor, MdiseaseColor = diseaseColor(img)
    BdiseaseArea, KdiseaseArea, MdiseaseArea = diseaseArea(img)
    Bdensity, Kdensity, Mdensity = density(img)

    pB = BcolorVariation*BdiseaseColor*BdiseaseArea*Bdensity
    pK = KcolorVariation*KdiseaseColor*KdiseaseArea*Kdensity
    pM = McolorVariation*MdiseaseColor*MdiseaseArea*Mdensity
    # pB = colorVariation(img)[0]*diseaseColor(img)[0]*diseaseArea(img)[0]*density(img)[0]
    # pK = colorVariation(img)[1]*diseaseColor(img)[1]*diseaseArea(img)[1]*density(img)[1]
    # pM = colorVariation(img)[2]*diseaseColor(img)[2]*diseaseArea(img)[2]*density(img)[2]

    max_prob = [pB,pK,pM]
    index = max_prob.index(max(max_prob))
    # print(pB,pK,pM)
    return  result_label[index],pB, pK, pM

# img = gt.getImage('images/upload/DSC_0130.JPG')
# # # # density(img)
# print(classification(img))
# print(colorVariation(img))
# print(diseaseColor(img))


