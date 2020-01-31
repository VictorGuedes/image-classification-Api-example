from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf

app = flask.Flask(__name__)
model = None


def load_model():
    global model
    model = MobileNetV2(weights="imagenet")


def prepare_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def predict_model(image):
    preds = model.predict(image)
    results = decode_predictions(preds, top=3)
    return results


def format_json(results):
    data = {}
    data["predictions"] = []
    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": float(prob)}
        data["predictions"].append(r)
    return data


@app.route("/predict", methods=["POST"])
def predict():

    if flask.request.files.get("image"):

        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image)

        results = predict_model(image)

        data = format_json(results)

    return flask.jsonify(data)

if __name__ == "__main__":
    load_model()

    app.run(host='0.0.0.0')