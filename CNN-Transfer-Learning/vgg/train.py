import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import argparse
from vgg import create_model, mkdir
from glob import glob
import os
import json


# Parse arguments
parser = argparse.ArgumentParser(description='Transfer learning using VGG')
parser.add_argument('--traindir', type=str,
                    help='train directory (labels are sub folders names)')
parser.add_argument('--validdir', type=str,
                    help='validation directory (labels are sub folders names)')
parser.add_argument('--epochs', type=int, help='number of epochs')
parser.add_argument('--image_size', nargs='+', help='image size e.g 100 100')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--modelprefix', type=str, help='directory to save model')
parser.add_argument('--finetune', type=bool, default = False, required=False,  help='fine tune or not')
parser.add_argument('--horizontal_flip', type=bool, default=False, required=False, help='horizontal flip invariant')
parser.add_argument('--vertical_flip', type=bool, default=False, required=False, help='vertial flip invariant')
parser.add_argument('--logdir', required=False, default='data\\logs\\')
parser.add_argument('--initial_epoch', type=int, required=False, default=0)

args = parser.parse_args()

args.traindir = args.traindir.rstrip('/\\')
args.validdir = args.validdir.rstrip('/\\')
args.image_size = [int(x) for x in args.image_size]
args.num_classes = len(glob(args.traindir + '/*'))
args.num_train_samples = len(glob(args.traindir + '/*/*.jp*g'))
args.num_valid_samples = len(glob(args.validdir + '/*/*.jp*g'))

mkdir(args.modelprefix)
mkdir(args.logdir)

print(args.horizontal_flip, args.vertical_flip, args.finetune)
# Prepare the input data

# Image data generator.
# Parameter here needs to be modified for different data set
# Invariant nature of data
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=args.horizontal_flip,
    vertical_flip=args.vertical_flip,
    preprocessing_function=preprocess_input
)

train_generator = gen.flow_from_directory(args.traindir,
                                          target_size=args.image_size,
                                          shuffle=True,
                                          batch_size=args.batch_size)

valid_generator = gen.flow_from_directory(args.validdir,
                                          target_size=args.image_size,
                                          shuffle=True,
                                          batch_size=args.batch_size)

# Make sure that two class_indices are exactly the same
assert train_generator.class_indices == valid_generator.class_indices

model = create_model(args.image_size, args.num_classes)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

filepath= args.modelprefix + "model-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=args.epochs,
    initial_epoch= args.initial_epoch,
    steps_per_epoch=args.num_train_samples // args.batch_size,
    validation_steps=args.num_valid_samples // args.batch_size,
    callbacks=[checkpoint, keras.callbacks.TensorBoard(args.logdir)]
)

model.save(args.modelprefix + 'model.h5')
with open(args.modelprefix + 'model.json', 'w') as f:
    f.write(model.to_json())
with open(args.modelprefix + 'model-classes.json', 'w') as f:
    f.write(json.dumps(train_generator.class_indices))
