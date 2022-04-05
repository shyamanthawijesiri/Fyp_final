import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import cv2
import os
from statistics import mean

import preprocessing as pre
import healthy_leaf as healthy_classification
import FK_holes as FK1
import FK_mutation as FK2

import scoring_model as model
import scoring_model_disease as disease_model



def cvtRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def resized(img):
    size = 512
    return cv2.resize(img,(size, size))



st.title('Betel vine Catergorization and Desease Detection')
fs = st.file_uploader('upload Image',['jpg','png','jpeg'])
save_path = 'images/upload/'

isDir = os.path.isdir(save_path)

if not isDir :
    os.makedirs(save_path,0o666)

if fs is not None:
    img_path = os.path.join(save_path, fs.name)
    with open(img_path,'wb') as f:
        f.write(fs.read())


    img = cv2.imread(img_path)
    resized_img = resized(img)
    display_img = cvtRGB(resized_img)
    # display_img = re
    # image = Image.open(img_path)
    # image2 = image.rotate(-90)

    # st.image(image2)
    if fs:
        col1, col2= st.columns(2)
        with col1:
            st.image(display_img)
        # healthy = 0
        with col2:
            input_img = os.path.join(save_path,fs.name)
            img = pre.getImage(input_img)
            bImg = pre.binaryImage(img)
            cImg, mask, leaf, cnt = pre.getLeaf(bImg, img)
            resized_leaf = resized(leaf)
            display_leaf = cvtRGB(resized_leaf)
            st.image(display_leaf)

        result = healthy_classification.healthyLeafClassification(leaf)
        if result == 0:
            new_title = '<p style="color:Green; font-size: 22px;">Healthy Leaf</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            print("healthy leaf")
            width, length, rMean, gMean, dif = model.featureExtraction(input_img)
            catergory, kalu, sudu, kanda, ran, kori = model.classification(width, length, rMean, gMean, dif)
            st.subheader("image name - {}".format(fs.name))
            st.subheader("Predicted category - {}".format(catergory))
            st.header("Probabilities")
            st.text("Kalu - {}".format(kalu))
            st.text("Sudu - {}".format(sudu))
            st.text("Kanda - {}".format(kanda))
            st.text("Ran - {}".format(ran))
            st.text("Korikan - {}".format(kori))
        else:
            print("unhealthy leaf")
            hole, hole_img = FK1.holes(leaf)
            mutation = FK2.mutation(cnt,mask)
            print(hole)
            if len(hole)>0 and mean(hole) > 160 :
                st.text("Kanamediri Haniya")
                st.image(hole_img)

            if mutation >=3:
                st.text("Kanamediri Haniya-mutation")
            else:
                print(img_path)
                disease_type, B, K, M = disease_model.classification(img_path)
                st.subheader("image name - {}".format(fs.name))
                st.subheader("Predicted category - {}".format(disease_type))
                st.header("Probabilities")
                st.text("Bacterial - {}".format(B))
                st.text("Kanamediri Haniya - {}".format(K))
                st.text("Betel Rust - {}".format(M))
#
