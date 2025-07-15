import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Load model
model = tf.keras.models.load_model('model/model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    label = "Dog" if prediction[0][0] >= 0.5 else "Cat"
    print(f"Predicted: {label} ({prediction[0][0]:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
