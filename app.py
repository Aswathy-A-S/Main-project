import streamlit as st
import cv2
import os
import numpy as np
from keras.models import load_model
from helper import prepare_image_for_ela, prerpare_img_for_weather
from fetchOriginal import image_coordinates, get_weather
from PIL import Image as PILImage
import io

# Constants
CLASS_WEATHER = ['Lightning', 'Rainy', 'Snow', 'Sunny']
CLASS_ELA = ['Real', 'Tampered']
OUTDOOR = False
ELA_MODEL_PATH = "ELA_Training/ela_model.h5"
WEATHER_MODEL_PATH = "WeatherCNNTraining/Weather_Model.h5"

# Load models once to avoid reloading in every function call
@st.cache_resource
def load_ela_model():
    return load_model(ELA_MODEL_PATH)

@st.cache_resource
def load_weather_model():
    return load_model(WEATHER_MODEL_PATH)

ELA_MODEL = load_ela_model()
WEATHER_MODEL = load_weather_model()

# Function to resize image for preview
def check_img(image_name):
    img = cv2.imread(image_name)
    img = cv2.resize(img, (750, 750))
    return img

# Function to detect tampering using ELA
def detect_ela(img_name):
    global ela_result
    np_img_input, ela_result = prepare_image_for_ela(img_name)
    Y_predicted = ELA_MODEL.predict(np_img_input, verbose=0)
    tamper_confidence = round(np.max(Y_predicted[0]) * 100)
    tamper_label = CLASS_ELA[np.argmax(Y_predicted[0])]
    return f"Model shows {tamper_confidence}% confidence of image being {tamper_label}", ela_result

# Function to predict weather conditions in the image
def detect_weather(img_name):
    np_img_input = prerpare_img_for_weather(img_name)
    Y_predicted = WEATHER_MODEL.predict(np_img_input, verbose=0)
    weather_label = CLASS_WEATHER[np.argmax(Y_predicted[0])]
    confidence = round(np.max(Y_predicted[0]) * 100)
    return f"Model shows weather in image as {weather_label} ({confidence}% confidence)"

# Function to fetch metadata-based weather validation
def org_weather(img_name):
    global OUTDOOR
    date_time, lat, long, OUTDOOR = image_coordinates(img_name)
    
    if not OUTDOOR:
        return "No outdoor metadata found (EXIF data might be masked)."
    
    location, date, weather = get_weather(date_time, lat, long)
    
    if lat == 0.0:
        return "Unable to fetch location (EXIF data masked)."
    
    if weather == "NA":
        return f"Image was taken at {location} on {date}; weather data unavailable."
    
    return f"Image taken at {location} on {date}. Historical weather: {weather}."

# Streamlit Interface
st.title("Image Tampering Detection Using ELA & Metadata Analysis")

st.markdown("""
### Detect Image Tampering & Validate Weather Metadata  
This tool uses **Error Level Analysis (ELA)** to detect image manipulation and **metadata-based weather validation** to cross-check the image’s recorded weather.
""")

# Image Upload Section
uploaded_file = st.file_uploader("Upload a JPEG image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Metadata-based outdoor validation option
    flag = st.radio("Does this image show an outdoor scene? (Requires metadata)", ('Yes', 'No'))

    if st.button("Analyze Image"):
        # Perform ELA analysis
        ela_result_text, ela_result_img = detect_ela(temp_path)

        # Perform metadata-based weather validation
        weather_metadata_result = ""
        weather_prediction_result = ""
        
        if flag == "Yes":
            weather_metadata_result = org_weather(temp_path)
            weather_prediction_result = detect_weather(temp_path)

        # Display results
        st.write("### Results:")
        st.write(f"1. {ela_result_text}")
        
        if flag == "Yes":
            if not OUTDOOR:
                st.write("2. Metadata analysis: **Unable to fetch location/time metadata.**")
            else:
                st.write(f"2. {weather_metadata_result}")
                st.write(f"3. {weather_prediction_result}")

        # Show ELA processed image
        st.image(ela_result_img, caption="ELA Processed Image")

        # Save ELA result image for download
        buffer = io.BytesIO()
        ela_result_img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Download button for processed ELA image
        st.download_button(
            label="Download ELA Processed Image",
            data=buffer,
            file_name="ela_result.jpg",
            mime="image/jpeg"
        )

    # Button to reset analysis
    if st.button("Try Another Image"):
        st.rerun()
