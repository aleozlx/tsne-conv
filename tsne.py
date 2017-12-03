import matplotlib.pyplot as plt

import os, sys
import itertools, functools
import numpy as np
import pandas as pd
import tensorflow as tf

import h5py
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Reshape, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

DATASET = lambda fname = '': os.path.join('/dsa/data/all_datasets/transfer_learning/DogsCats', fname)
assert os.path.exists(DATASET())

class DogsVsCats(Model):
    def __init__(self):
        self.images = Input(shape = [150, 150, 3])
        self.vgg16 = VGG16(weights = None, include_top = False)
        
        classifier = [
            Flatten(input_shape = self.vgg16.output_shape[1:]),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ]

        self.prediction = functools.reduce(lambda f1, f2: f2(f1), [self.images, self.vgg16]+classifier)
        
        super(DogsVsCats, self).__init__(
            inputs = [self.images],
            outputs = [self.prediction]
        )
        
        self.compile(loss='binary_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])
        
    def freeze_vgg16(trainable = False):
        for layer in self.vgg16.layers:
            layer.trainable = trainable

y_pred = functools.reduce(lambda f1, f2: f2(f1), [images, vgg16]+classifier)

model = DogsVsCats()
# model.load_weights('./weights_dogs_cats.h5')
BATCH_SIZE = 20

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    DATASET('TransferLearning/train'),
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    DATASET('TransferLearning/validation'),
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary')

#model.fit_generator(
#        train_generator, steps_per_epoch = 12500 // BATCH_SIZE,
#        validation_data=validation_generator, validation_steps=800 // BATCH_SIZE,
#        epochs=1)

#model.save_weights('./weights_dogs_cats.h5')

loss, accuracy = model.evaluate_generator(validation_generator, steps = 800 // BATCH_SIZE)
print('loss:', loss, 'accuracy:', accuracy)

#Here's how you can make a prediction for one image using this model.
#im_test = imread(DATASET('test1/5.jpg'))
#imshow(im_test)
#im_test = resize(im_test, (150, 150), mode = 'reflect')
#y_pred = new_model.predict(np.expand_dims(im_test, 0)).squeeze()
#print(['Cat', 'Dog'][y_pred>=0.5])

