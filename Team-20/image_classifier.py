import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess

def classify_image(test_image, model_type):
    width, height = 224, 224
    model = load_model(model_type)

    if model_type == "cervical_dl.h5":
        feature_extractor = ResNet50(weights='imagenet', include_top=False)
        preprocess_input = resnet_preprocess
    elif model_type == "mobnet.h5":
        feature_extractor = MobileNet(weights='imagenet', include_top=False)
        preprocess_input = mobilenet_preprocess
    elif model_type == "incep.h5":
        feature_extractor = InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = inception_preprocess
    else:
        raise ValueError("Invalid model_type provided.")

    resized_test_image = cv2.resize(test_image, (width, height))
    expanded_test_image = np.expand_dims(resized_test_image, axis=0)
    preprocessed_test_image = preprocess_input(expanded_test_image)

    features = feature_extractor.predict(preprocessed_test_image)
    features = features.reshape(features.shape[0], -1)

    y_pred = model.predict(features)
    decoded_predictions = np.argmax(y_pred, axis=1)
    class_probabilities = np.max(y_pred, axis=1)
    class_labels = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
    predicted_class_index = np.argmax(y_pred[0])
    predicted_probability = np.max(y_pred[0])

    return class_labels[predicted_class_index], predicted_probability

def classify_image_interface(image, model_type):
    image = image.convert("RGB")
    test_image = np.array(image)
    predicted_class, predicted_probability = classify_image(test_image, model_type)
    return f"Predicted Class: {predicted_class}\nPredicted Probability: {predicted_probability:.2f}"
