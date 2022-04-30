import streamlit as st
import streamlit.components.v1 as components
import time

from PIL import Image
import cv2
import os
import operator
from statistics import mean

import preprocessing as pre
import healthy_leaf as healthy_classification
import FK_holes as FK1
import FK_mutation as FK2

import scoring_model as model
import scoring_model_disease as disease_model
import disease_percentage as dp



def cvtRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def resized(img):
    size = 512
    return cv2.resize(img,(size, size))

st.title('Betel vine Categorization and Disease Detection')
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
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")

        with col2:
            st.image(display_img)

        with col3:
            st.write("")

        input_img = os.path.join(save_path,fs.name)
        img = pre.getImage(input_img)
        bImg = pre.binaryImage(img)
        cImg, mask, leaf, cnt = pre.getLeaf(bImg, img)
        # resized_leaf = resized(leaf)
        # display_leaf = cvtRGB(resized_leaf)
        # st.image(display_leaf)

        result = healthy_classification.healthyLeafClassification(leaf)
        if result == 0:
            with st.spinner('Wait for it...'):
                time.sleep(2)
            new_title = '<p style="color:Green; font-size: 32px;font-weight:bold; text-align: center;">Healthy Leaf</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            print("healthy leaf")
            width, length, rMean, gMean, gStd, dif = model.featureExtraction(input_img)
            scores = model.classification(width, length, rMean, gMean,gStd, dif)
            category = max(scores.items(), key=operator.itemgetter(1))[0]
            # st.subheader("image name - {}".format(fs.name))

            col1, col2, col3= st.columns([1, 4, 3])

            with col1:
                st.write("")
            with col2:
                with st.spinner('Wait for it...'):
                    time.sleep(4)
                st.subheader("Predicted category")
                new_title = f'<p style="color:red; font-size: 32px; font-weight:bold;">{category}</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                # st.subheader("Predicted category - {}".format(category))
            with col3:
                st.subheader("Probabilities")
                for c,s in scores.items():
                    st.text("{} - {}".format(c,s))

        else:
            with st.spinner('Wait for it...'):
                time.sleep(2)
            new_title = '<p style="color:red; font-size: 32px; font-weight:bold; text-align: center;">Unhealthy Leaf</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            print("unhealthy leaf")
            hole, hole_img,K_percentage = FK1.holes(leaf)
            mutation = FK2.mutation(cnt,mask)
            if len(hole)>0 and mean(hole) > 160 :
                col1, col2, col3 = st.columns([1, 4, 3])

                with col1:
                    st.write("")
                with col2:
                    with st.spinner('Wait for it...'):
                        time.sleep(4)
                    st.subheader("Predicted Disease")
                    KH = f'<p style="color:red; font-size: 22px; font-weight:bold;">Kanamediri Haniya</p>'
                    st.markdown(KH, unsafe_allow_html=True)
                    # st.subheader("Predicted category - {}".format(category))
                with col3:
                    st.subheader("Disease Percentage")
                    per = f'<p style="color:purple; font-size: 22px; font-weight:bold;">{round(K_percentage,2)} %</p>'
                    st.markdown(per, unsafe_allow_html=True)


            elif mutation >=3:
                col1, col2, col3 = st.columns([1, 4, 3])

                with col1:
                    st.write("")
                with col2:
                    with st.spinner('Wait for it...'):
                        time.sleep(4)
                    st.subheader("Predicted Disease")
                    KH = f'<p style="color:red; font-size: 22px; font-weight:bold;">Kanamediri Haniya</p>'
                    st.markdown(KH, unsafe_allow_html=True)
                    # st.subheader("Predicted category - {}".format(category)

            else:
                col1, col2, col3 = st.columns([1, 4, 3])
                with col1:
                    st.write("")
                with col2:
                    with st.spinner('Wait for it...'):
                        time.sleep(4)
                    disease_type, B, K, M = disease_model.classification(img_path)
                    percentage = dp.percentage(img_path)
                    st.subheader("Predicted Disease")
                    disease = f'<p style="color:red; font-size: 22px; font-weight:bold;">{disease_type}</p>'
                    st.markdown(disease, unsafe_allow_html=True)
                    st.subheader("Disease Percentage")
                    per = f'<p style="color:purple; font-size: 22px; font-weight:bold;">{round(percentage,2)} %</p>'
                    st.markdown(per, unsafe_allow_html=True)
                    # st.subheader("Predicted category - {}".format(category))
                with col3:
                    st.subheader("Probabilities")
                    st.text("Bacterial - {}".format(round(B,4)))
                    st.text("Kanamediri Haniya - {}".format(round(K,4)))
                    st.text("Betel Rust - {}".format(round(M,4)))

