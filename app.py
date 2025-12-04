import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- App Title ---
st.title("😊 Image Classification App (Happy / Sad)")
st.write("Upload an image and let the model predict your mood!")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models.h5")  # <-- make sure model.h5 is in same folder
    return model

model = load_model()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 200.0  # rescale same as training

    # Predict
    prediction = model.predict(img_array)
    result = "Happy 😊" if prediction[0][0] < 0.5 else "Sad 😔"

    # Show result
    st.subheader("Prediction Result:")
    st.success(result)
else:
    st.info("Please upload an image to get prediction.")