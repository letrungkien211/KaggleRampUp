
import numpy as np
from PIL import Image
from flask import Flask, abort, request, jsonify
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import img_to_array

import json
import io

import argparse

def load_index2labels(classes_indices):
    with open(classes_indices) as f:
        label2index = json.loads(f.read())
        index2label = [None]* len(label2index)
        for k, v in label2index.items():
            index2label[v] = k
    return index2label

def load(model_path):
    model = load_model(model_path)
    model._make_predict_function()
    image_size = model.layers[0].get_output_at(0).get_shape().as_list()[1:3]
    return (model, image_size)

def prepare_image(img, img_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def create_app():
    app = Flask(__name__)

    @app.route('/keepalive')
    def keepalive():
        return 'I am alive!'

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return abort(400)
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img))
        img = prepare_image(img, IMAGE_SIZE)

        preds = MODEL.predict(img)
        label_index = preds.argmax(axis=-1).item(0)
        label = INDEX_TO_LABELS[label_index]
        ret = {'label': label, 'prob': preds.item(label_index)}
        return jsonify(ret)
    return app

def init():
    parser = argparse.ArgumentParser(description='Running server')
    parser.add_argument('--model', required=True)
    parser.add_argument('--classes_indices', required=True)

    args = parser.parse_args()
    global MODEL, IMAGE_SIZE, INDEX_TO_LABELS
    MODEL, IMAGE_SIZE = load(args.model)
    INDEX_TO_LABELS = load_index2labels(args.classes_indices)
    
if __name__ == "__main__":
    print('Loading keras model and flask')
    init()
    app = create_app()
    app.run(host='0.0.0.0')    