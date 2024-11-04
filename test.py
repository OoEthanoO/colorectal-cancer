import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize_with_pad
from PIL import Image
import sys
import json

model = load_model('best_transfer_cnn.h5')

with open('label_names.json', 'r') as f:
    label_names = json.load(f)

def preprocess_image(image_path, target_height, target_width):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = resize_with_pad(image, target_height, target_width, antialias=True)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image_path):
    image = preprocess_image(image_path, 224, 224)
    prediction = model.predict(image)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label_name = label_names[predicted_label_index]
    return predicted_label_name

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predicted_label = predict_image(image_path)
    print(f"Predicted Label: {predicted_label}")
