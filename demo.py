import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('OG.pkl')

# Streamlit app code to make predictions
st.title("Fake Logo Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize and prepare the image for prediction
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)

    # Display result
    if prediction[0] > 0.5:
        st.write("This is a **Fake** logo!")
    else:
        st.write("This is a **Real** logo!")
#streamlit run your_streamlit_script.py

