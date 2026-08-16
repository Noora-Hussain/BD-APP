import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 
st.title("Bahraini Currency Recognition")
 
model = tf.keras.models.load_model("currency_5model.keras")
 
class_names = ["0.05", "0.100", "0.25", "0.5 BD", "0.50",
               "1BD", "5 BD", "10 BD", "20 BD"]
 
uploaded_image = st.file_uploader(
    "Upload a mobile-phone photo",
    type=["jpg", "jpeg", "png"])
 
camera_image = st.camera_input("Or take a live-camera photo")
 
image_file = uploaded_image
if camera_image is not None:
    image_file = camera_image
 
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    width, height = image.size
 
    # The model resizes down to just 64x64 pixels. If the currency only
    # fills a small part of a larger photo (e.g. a coin on a whiteboard
    # with lots of empty space around it), most of what survives that
    # resize is background, and the coin/note itself can shrink to only a
    # handful of pixels - not enough detail left to tell currencies apart.
    # This crop box lets you tighten the photo around the currency before
    # it's resized, so the model sees mostly currency instead of mostly
    # background.
    st.write("Drag the sliders so the box tightly surrounds the currency:")
    left, right = st.slider("Horizontal crop (%)", 0, 100, (25, 75))
    top, bottom = st.slider("Vertical crop (%)", 0, 100, (25, 75))
 
    box = (
        int(width * left / 100),
        int(height * top / 100),
        int(width * right / 100),
        int(height * bottom / 100))
    cropped = image.crop(box)
 
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original")
    col2.image(cropped, caption="Cropped (what the model actually sees)")
 
    model_input = cropped.resize((64, 64))
    model_input = np.array(model_input) / 255
    model_input = np.expand_dims(model_input, axis=0)
 
    prediction = model.predict(model_input)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
 
    st.success(
        f"Predicted currency: {class_names[predicted_class]} "
        f"({confidence:.0%} confidence)")
 
    with st.expander("Show full prediction breakdown"):
        probs = {class_names[i]: float(prediction[0][i])
                 for i in range(len(class_names))}
        st.bar_chart(probs)
