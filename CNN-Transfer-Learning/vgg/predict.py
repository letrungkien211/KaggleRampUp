from tensorflow import keras
from keras.models import load_model, Model
from keras.preprocessing import image
import numpy as np
import json
from keras.applications.vgg16 import VGG16, preprocess_input
import os


import argparse
parser = argparse.ArgumentParser(description='Transfer learning using VGG')
parser.add_argument('--model', help='model path')
parser.add_argument('--classes', help='class mapping')
parser.add_argument('--image', help='image path')

args = parser.parse_args()

model = load_model(args.model)
model.summary()

with open(args.classes) as f:
    label2index = json.loads(f.read())
    index2label = [None]* len(label2index)
    for k, v in label2index.items():
        index2label[v] = k

input_shape = model.layers[0].get_output_at(0).get_shape().as_list()[1:3]
img = image.load_img(args.image, target_size=input_shape)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(x.shape)

probs = model.predict(x)
label_index = probs.argmax(axis=-1)
print(probs)
print(index2label)
print(index2label[label_index.item(0)])
