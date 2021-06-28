import os
import io
import pandas as pd
import numpy as np
import string
import random
import PIL
from PIL import Image
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from urllib.request import urlopen
from io import BytesIO
from tensorflow import keras
from skimage.transform import resize
import tensorflow as tf

tf.keras.backend.clear_session()


def main():
    model_tensor = keras.models.load_model('C:\DS\Freecodecamp\Dashboard/model_tensor.h5')
    tf.keras.backend.clear_session()
    def predict_image(image_upload, model = model_tensor):
        im = Image.open(image_upload)
        resized_im = im.resize((224, 224))
        im_array = np.asarray(resized_im)
        im_array = im_array*(1/225)
        im_input = tf.reshape(im_array, shape = [1, 224, 224, 3])

        new_predict = model.predict(im_input)
        list_index = [0,1,2]

        x = new_predict
        for i in range(3):
            for j in range(3):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp
        #show sorted labels in order
        if list_index[0] == 0:
            prediction_prod = 'the uploaded image is classified as folding bikes'
        elif list_index[0] == 1:
            prediction_prod = 'the uploaded image is classified as kids bikes'

        else:
            prediction_prod = 'the uploaded image is classified as mountain bikes'

        return prediction_prod, im

    st.sidebar.title('Navigation')
    pages = st.sidebar.radio("Pages", ("Home Page", "Image Classifier"))

    if pages == "Home Page":
        st.write("""# Bike image classification""")
        st.markdown(""" The dataset was collected manually from google images searches and it is around 300 images""")
        st.markdown("""The model used in this project is Convolutional Neural Network in Tensorflow""")
        st.markdown("""The final model that is used in this project is using VGG16 that is modified to predict 3 classes """)
        st.markdown(""" """)
    elif pages == "Image Classifier":
        st.title("Image Classifier")
        
        image_url = st.text_input("Paste the image file's link here")
        if st.button("Classify the image"):

            file = BytesIO(urlopen(image_url).read())
            img = file
            label, uploaded_image = predict_image(img)
            st.image(uploaded_image, width = None)
            st.write(label)

if __name__ == '__main__':
    main()



