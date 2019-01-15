
# import numpy as np
# from PIL import Image
# from flask import Flask, abort, request, jsonify
# from keras.models import Model, load_model
# from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils

# python  server.py  --model_path  ./data/inference_data/model-weights-36-0.62.hdf5  --image_size  224  224  --classes_indices  ./data/inference_data/model-classes.json

import numpy as np
from PIL import Image
from flask import Flask, abort, request, jsonify
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Lambda, GlobalAveragePooling2D
import os
import keras
import tensorflow as tf
import json
import io

import argparse

print('keras', keras.__version__)
print('tensorflow', tf.VERSION)

parser = argparse.ArgumentParser(description='Running server')
parser.add_argument('--model_path', required=True)
parser.add_argument('--classes_indices', required=True)
parser.add_argument('--image_size', nargs='+', default=(224,224), required=False)
parser.add_argument('--port', type=int, required=False, default=5000)

args = parser.parse_args()

args.image_size = [int(x) for x in args.image_size]

def FeaturesExtractor(input_shape, pretrained_model, preprocess_input):
    model = pretrained_model(include_top=False, input_shape=input_shape, weights='imagenet')
    inputs = Input(input_shape)
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = model(x)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs, x)

def load_index2labels(classes_indices):
    with open(classes_indices) as f:
        label2index = json.loads(f.read())
        index2label = [None]* len(label2index)
        for k, v in label2index.items():
            index2label[v] = k
    return index2label

def get_input_shape_as_list(model):
    return model.layers[0].get_output_at(0).get_shape().as_list()[1:]

def prepare_image(img, img_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
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
        img = FEATURE_EXTRACTOR.predict(img)
        preds = MODEL.predict(img)
        label_index = preds.argmax(axis=-1).item(0)
        label = INDEX_TO_LABELS[label_index]
        ret = {'label': label, 'prob': preds.item(label_index)}
        return jsonify(ret)
    return app

def init():
    if(args.port==5000):
        print('skip init')
        return
    print('1')
    global MODEL, IMAGE_SIZE, INDEX_TO_LABELS, FEATURE_EXTRACTOR
    MODEL = load_model(args.model_path)
    print('2')
    MODEL._make_predict_function()
    print('3')
    FEATURE_EXTRACTOR = FeaturesExtractor(args.image_size+[3,], VGG16, preprocess_input)
    print('4')
    FEATURE_EXTRACTOR._make_predict_function()
    print('5')
    IMAGE_SIZE = args.image_size
    print('6')
    INDEX_TO_LABELS = load_index2labels(args.classes_indices)
    print('7')

    
if __name__ == "__main__":
    print('Loading keras model and flask')
    init()
    print('Create flask app')
    app = create_app()
    print('Run the app')
    app.run(host='0.0.0.0', port=args.port)
    print('Exit')