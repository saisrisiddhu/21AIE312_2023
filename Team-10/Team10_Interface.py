import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load the segmentation model
segmentation_model = tf.keras.models.load_model('segmmodel.h5', compile=False)

# Define the target size for segmentation
segmentation_target_size = (512, 512)

# Function to preprocess the image for segmentation
def preprocess_image_segmentation(image):
    original_image = image.copy()  # Make a copy of the original image
    image = image.resize(segmentation_target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image, original_image

# Function to resize the segmentation mask to the original image size
def resize_segmentation(segmentation_mask, original_size):
    resized_mask = cv2.resize(segmentation_mask, original_size, interpolation=cv2.INTER_NEAREST)
    return resized_mask

# Configure Streamlit
st.title("Image Segmentation")
st.write("Upload an image and the model will perform segmentation.")

# Image upload and prediction
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for segmentation
    seg_img, original_image = preprocess_image_segmentation(img)

    # Perform segmentation
    seg_prediction = segmentation_model.predict(seg_img)
    seg_mask = np.argmax(seg_prediction, axis=-1)

    # Resize the segmentation mask to the original image size
    resized_seg_mask = resize_segmentation(seg_mask[0], original_image.size)

    # Display the segmentation mask
    st.subheader("Segmentation Mask")
    plt.imshow(resized_seg_mask, cmap='gray')
    plt.axis('off')
    st.pyplot(plt)
