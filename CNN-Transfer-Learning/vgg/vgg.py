
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
import os

def create_model(image_size, num_classes, fine_tune = False):
    vgg = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)
    if not fine_tune:
        for layer in vgg.layers:
            layer.trainable = False
    x = Flatten()(vgg.output)
    prediction = Dense(num_classes, activation='softmax')(x)
    return Model(inputs= vgg.input, outputs=prediction)

def mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)