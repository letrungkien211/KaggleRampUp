
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow import keras
import argparse
import os
parser = argparse.ArgumentParser(description='Transfer learning using VGG')
parser.add_argument('--model', required=True, help='model path')
parser.add_argument('--classes', required=True, help='class mapping')
parser.add_argument('--image', required=True, help='image path')
parser.add_argument('--usecpu', required=False, default=False)

args = parser.parse_args()

if(args.usecpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


model = load_model(args.model)
model.summary()
model._make_predict_function()

with open(args.classes) as f:
    label2index = json.loads(f.read())
    index2label = [None] * len(label2index)
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
