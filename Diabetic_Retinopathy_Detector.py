import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
import time
# Load the pre-trained model

def set_css():
    """
    Function to set the CSS style of the app.
    """
    st.markdown("""
    <style>
    .stApp {
        background-color: #295953;
    }
    .stButton button {
        background-color: #c9ffee;
        color: #C70039;
    }
    </style>
    """, unsafe_allow_html=True)
def main():
    st.title("Diabetic Retinopathy Detector")
    st.subheader("By ABDULWAHAB BALQEES")
    st.subheader("21D/47CS/01596")
    uploaded_image = st.file_uploader("Upload Retinopathy Image To Get Result(No DR, Mild, Moderate, Severe, Proliferative DR)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.container():
            st.write('<style>div.stImage>img {display:block;margin-left:auto;margin-right:auto;}</style>', unsafe_allow_html=True)
            st.image(uploaded_image, caption="Uploaded Retinopathy Image",width=400,use_column_width=False)
            image = Image.open(uploaded_image)
            model_path='cnn_model_final.h5'
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=["acc"])
            img = image.resize((128, 128))
            img = np.array(img)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image=cv2.addWeighted ( image,6,cv2.GaussianBlur( image , (0,0) , 10),-6 ,128)
            image = cv2.medianBlur(image, 1)
            img = image/ 255.0
            img = img.reshape((1, 128, 128, 3))
    
    if st.button("Submit"):
        if uploaded_image is not None:
            pred=model.predict(img)
            class_idx = np.argmax(pred)
    # Predict the image class
            if class_idx== 0:
                with st.spinner('Loading the Results...'):
                    time.sleep(1)
                st.success('No DR')
            elif class_idx== 1:
                with st.spinner('Loading the Results....'):
                    time.sleep(1)
                st.success('Mild')
            elif class_idx== 2:
                with st.spinner('Loading the Results...'):
                    time.sleep(1)
                st.success('Moderate')
            elif class_idx== 3:
                with st.spinner('Loading the Results...'):
                    time.sleep(1)
                st.success('Severe')
            elif class_idx== 4:   
                with st.spinner('Loading the Results...'):
                    time.sleep(1)
                st.success('Proliferative DR')         
            
        else:
            st.warning("Please upload an Retinopathy Image to view your results")
            
if __name__ == '__main__':
    set_css()
    main()


