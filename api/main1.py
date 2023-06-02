from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

MODEL = tf.keras.models.load_model("C:\\Users\\kalle\\PPDC\\Potato-Disease-Classification-master\\saved_models\\potatoes.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

FERTILIZER_RECOMMENDATIONS = {
    'Early Blight': {
        'Balanced Fertilizers': ['Using a balanced fertilizer with equal or similar amounts of nitrogen (N), phosphorus (P), and potassium (K) can help promote overall plant health without excessively stimulating foliage growth, which can make plants more susceptible to Early Blight. Look for fertilizers with an N-P-K ratio of 10-10-10 or similar.'],
        'Organic Fungicides': ['Copper-based products - Early Blight is caused by a fungus, so using organic fungicides like copper-based products can help suppress its spread. While not a fertilizer, it is important in managing the disease.']
    },
    'Late Blight': {
        'Potassium-rich Fertilizers': ['Late Blight management often focuses on promoting plant vigor and disease resistance. Potassium (K) is particularly beneficial in this regard. Fertilizers with a higher ratio of potassium, such as those with an N-P-K ratio of 5-10-10 or similar, can help improve plant health and resistance to Late Blight.'],
        'Organic Matter': ['Incorporating organic matter, such as compost or well-rotted manure, into the soil can improve overall plant health and enhance disease resistance. It promotes beneficial microbial activity and supports the plants natural defense mechanisms.']
    },
    'Healthy': {
        'Nitrogen-based Fertilizers': ['Ammonium Nitrate - Nitrogen (N) is essential for plant growth and development. Using a nitrogen-rich fertilizer, such as ammonium nitrate or urea, can promote healthy foliage and overall plant vigor. However, avoid excessive nitrogen application, as it can increase susceptibility to certain diseases.'],
        'Balanced Fertilizers': ['Using a balanced fertilizer with equal or similar amounts of nitrogen (N), phosphorus (P), and potassium (K) can provide the necessary nutrients for overall plant health. Look for fertilizers with an N-P-K ratio of 10-10-10 or similar.'],
        'Organic Matter': ['Incorporating organic matter into the soil, such as compost or well-rotted manure, improves soil fertility, structure, and nutrient availability. It also supports beneficial microbial activity, which enhances overall plant health']
    }
}



app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")


@app.route("/ping", methods=["GET"])
def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(
        Image.open(BytesIO(data)).convert(
            "RGB").resize((256, 256))  # image resizing
    )
    return image


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Get Fertilizer Recommendation based on the identified disease
    fertilizer_recommendation = FERTILIZER_RECOMMENDATIONS.get(predicted_class)

    return jsonify({
        'class': predicted_class,
        'confidence': confidence,
        'fertilizer': fertilizer_recommendation
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
