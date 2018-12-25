from tensorflow import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model 
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np 
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Transfer learning using VGG')
parser.add_argument('traindir', type=str, help='train directory')
parser.add_argument('testdir', type=str, help='test directory')
parser.add_argument('epoch', type=int, help='number of epoch')

