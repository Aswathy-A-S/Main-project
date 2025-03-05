import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from helper import prepare_image_for_ela
from PIL import Image as PILImage
import io

# Constants 
class_ELA = ['Real', 'Tampered'] 

# Function to perform ELA detection
def detect_ELA(img_name):
    np_img_input, ela_result = prepare_image_for_ela(img_name)
    ELA_model = load_model('ELA_Training/ela_model.h5')
    Y_predicted = ELA_model.predict(np_img_input, verbose=0)
    prediction = "Model shows {}% accuracy of image being {}".format(
        round(np.max(Y_predicted[0]) * 100), 
        class_ELA[np.argmax(Y_predicted[0])]
    )
    return prediction, ela_result

# Streamlit Interface
st.title("Error Level Analysis (ELA) - Image Tampering Detection")

st.markdown("""
Upload an image, and the system will perform Error Level Analysis to detect if the image has been tampered with.
""")

# Handle file upload
uploaded_file = st.file_uploader("Choose a .jpg/.jpeg image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Run ELA detection
    if st.button("Run Error Level Analysis"):
        res1, ela_result = detect_ELA("temp.jpg")
        st.write("### ELA Detection Result:")
        st.write(res1)

        # Display the ELA result
        st.image(ela_result, caption="ELA Processed Image")

        # Save ELA result image to a BytesIO object for downloading
        buffer = io.BytesIO()
        ela_result.save(buffer, format="JPEG")  
        buffer.seek(0)

        # Add a download button for the ELA result image
        st.download_button(
            label="Download ELA Result Image",
            data=buffer,
            file_name="ela_result.jpg",
            mime="image/jpeg"
        )
