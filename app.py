import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model



st.title("Tomato Leaf Classification")
st.header("Please upload the image to be classified:")
st.text("Created by SU")

#@st.cache(allow_output_mutation=True)
#def teachable_machine_classification(img, weights_file):
    
uploaded_file = st.file_uploader("Select an Image...", type="jpg")

# Load the model
model = tf.keras.models.load_model("saved_tomato2_plus_checkpts_wandb.h5")
opt = Adam(learning_rate= 0.0001)
model.compile(optimizer=opt, loss= 'categorical_crossentropy', metrics=['accuracy'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

#     res = model.evaluate(data)
#     st.write ("Loss and accuracy are:" + str(res))
    
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
    if prediction[0][0] == 1:
        st.write ("Predicted class is: Bacterial Spot")
    elif prediction[0][1] == 1:
        st.write ("Predicted class is: Early Blight")
    elif prediction[0][2] == 1:
        st.write ("Predicted class is: Late Blight")
    elif prediction[0][3] == 1:
        st.write ("Predicted class is: Leaf Mold")
    elif prediction[0][4] == 1:
        st.write ("Predicted class is: Septoria Leaf Spot")
    elif prediction[0][5] == 1:
        st.write ("Predicted class is: Target Spot")
    elif prediction[0][6] == 1:
        st.write ("Predicted class is: Mosaic Virus")
    elif prediction[0][7] == 1:
        st.write ("Predicted class is: Yellow Leaf Curl Virus")
    elif prediction[0][8] == 1:
        st.write ("Predicted class is: Two Spotted Target Mkites")
    elif prediction[0][9] == 1:
        st.write ("Predicted class is: Healthy")
    
    
    st.write ("Predictions are:", prediction)
    
    st.write ("Prediction percentage:", prediction_percentage)
    
    #return  prediction,prediction_percentage
