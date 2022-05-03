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
heading = f'<h1 style="font-weight:bold;text-align:center;">Betel vine Leaves Categorization and Disease Detection</h1>'
st.markdown(heading, unsafe_allow_html=True)
# st.title('Betel vine Leaves Categorization and Disease Detection')
fs = st.file_uploader('upload Image',['jpg','png','jpeg'])
save_path = 'images/upload/'

isDir = os.path.isdir(save_path)
t=2
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

            col1, col2, col3= st.columns([1, 3, 4])

            with col1:
                st.write("")
            with col2:
                with st.spinner('Wait for it...'):
                    time.sleep(t)
                heading = f'<h4 style="font-weight:bold;">Predicted Category</h4>'
                st.markdown(heading, unsafe_allow_html=True)

                heading = f'<h4 style="font-weight:bold;">Scores</h4>'
                st.markdown(heading, unsafe_allow_html=True)
                for index,(c,s) in enumerate(scores.items()):
                    st.text("{}.{} - {}".format(index+1,c,round(s,4)))
            with col3:
                cat = f'<p style="color:#4169e1; font-size: 30px; font-weight:bold; margin-top:5px;">{category}</p>'
                st.markdown(cat, unsafe_allow_html=True)


        else:
            with st.spinner('Wait for it...'):
                time.sleep(2)
            new_title = '<p style="color:red; font-size: 32px; font-weight:bold; text-align: center;">Unhealthy Leaf</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            print("unhealthy leaf")
            hole, hole_img,K_percentage = FK1.holes(leaf)
            mutation = FK2.mutation(cnt,mask)
            if len(hole)>0 and mean(hole) > 160 :
                with st.spinner('Wait for it...'):
                    time.sleep(t)
                col1, col2, col3 = st.columns([1, 3, 4])
                with col1:
                    st.write("")
                with col2:
                    heading = f'<h4 style="font-weight:bold;">Predicted Disease</h4>'
                    st.markdown(heading, unsafe_allow_html=True)
                    heading = f'<h4 style="font-weight:bold;">Disease Percentage</h4>'
                    st.markdown(heading, unsafe_allow_html=True)

                    # st.subheader("Predicted Disease")
                    # KH = f'<p style="color:red; font-size: 22px; font-weight:bold;">Kanamediri Haniya</p>'
                    # st.markdown(KH, unsafe_allow_html=True)
                    # st.subheader("Predicted category - {}".format(category))
                with col3:
                    disease = f'<p style="color:#4169e1; font-size: 20px; font-weight:bold; margin-top:14px;">Kalamediri Haniya</p>'
                    st.markdown(disease, unsafe_allow_html=True)
                    per = f'<p style="color:#4169e1; font-size: 20px; font-weight:bold; margin-top:18px;">{round(K_percentage, 2)} %</p>'
                    st.markdown(per, unsafe_allow_html=True)
                    #
                    # st.subheader("Disease Percentage")
                    # per = f'<p style="color:purple; font-size: 22px; font-weight:bold;">{round(K_percentage,2)} %</p>'
                    # st.markdown(per, unsafe_allow_html=True)

            elif mutation >=3:
                with st.spinner('Wait for it...'):
                    time.sleep(t)
                col1, col2, col3 = st.columns([1, 3, 4])

                with col1:
                    st.write("")
                with col2:
                    heading = f'<h4 style="font-weight:bold;">Predicted Disease</h4>'
                    st.markdown(heading, unsafe_allow_html=True)
                    # st.subheader("Predicted Disease")
                    # KH = f'<p style="color:red; font-size: 22px; font-weight:bold;">Kanamediri Haniya</p>'
                    # st.markdown(KH, unsafe_allow_html=True)
                    # st.subheader("Predicted category - {}".format(category)
                with col3:
                    disease = f'<p style="color:#4169e1; font-size: 20px; font-weight:bold; margin-top:14px;">Kalamediri Haniya</p>'
                    st.markdown(disease, unsafe_allow_html=True)
            else:
                with st.spinner('Wait for it...'):
                    time.sleep(t)
                col1, col2,col3 = st.columns([1, 3, 4])
                disease_type, B, K, M = disease_model.classification(img_path)
                percentage = dp.percentage(img_path)
                with col1:
                    st.write("")
                with col2:
                    heading = f'<h4 style="font-weight:bold;">Predicted Disease</h4>'
                    st.markdown(heading, unsafe_allow_html=True)

                    # st.subheader("Predicted Disease ")

                    heading = f'<h4 style="font-weight:bold;">Disease Percentage</h4>'
                    st.markdown(heading, unsafe_allow_html=True)

                    # st.subheader("Disease Percentage ")

                    heading = f'<h4 style="font-weight:bold;">Scores</h4>'
                    st.markdown(heading, unsafe_allow_html=True)
                    # st.subheader("Probabilities")
                    st.text("1.Bacterial Leaf Blight - {}".format(round(B, 4)))
                    st.text("2.Kalamadiri Haniya - {}".format(round(K, 4)))
                    st.text("3.Betel Rust - {}".format(round(M, 4)))
                    # st.subheader("Predicted category - {}".format(category))
                with col3:
                    disease = f'<p style="color:#4169e1; font-size: 20px; font-weight:bold; margin-top:14px;">{disease_type}</p>'
                    st.markdown(disease, unsafe_allow_html=True)
                    per = f'<p style="color:#4169e1; font-size: 20px; font-weight:bold; margin-top:18px;">{round(percentage, 2)} %</p>'
                    st.markdown(per, unsafe_allow_html=True)


