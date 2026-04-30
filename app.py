from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

import os
import gdown

MODEL_PATH = "plant_disease_recog_model_pwp.keras"

# download from Drive if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?export=download&id=1M2qYTbYFxWqwvE3QbTbUsKKx05Mh-9DQ",
        MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (160, 160)

class_names = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
'Apple___healthy','Background_without_leaves','Blueberry___healthy',
'Cherry___Powdery_mildew','Cherry___healthy',
'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust',
'Corn___Northern_Leaf_Blight','Corn___healthy','Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
'Peach___healthy','Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy','Potato___Early_blight',
'Potato___Late_blight','Potato___healthy','Raspberry___healthy',
'Soybean___healthy','Squash___Powdery_mildew',
'Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight',
'Tomato___Late_blight','Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "disease": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
