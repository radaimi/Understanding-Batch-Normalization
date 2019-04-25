from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.utils import np_utils
from keras.layers import Layer
import os
import keras.backend as K
import numpy as np

BATCH_NORM = True
NOISE = True
batch_size =  32
num_classes = 10
epochs = 100


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test_samples')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

class NoiseLayer(Layer):

    def __init__(self, mean, stddev, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev
        self.mean = mean

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs), mean=self.mean, stddev = self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'mean': self.mean, 'stddev': self.stddev}
        base_config = super(NoiseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def base_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
    model.add(BatchNormalization()) if BATCH_NORM else None
    #model.add(NoiseLayer(0.13, 0.5)) if NOISE else None
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization()) if BATCH_NORM else None
    #model.add(NoiseLayer(0.1, 0.3)) if NOISE else None
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    #model.add(NoiseLayer(0.2, 0.1)) if NOISE else None
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization()) if BATCH_NORM else None
    #model.add(NoiseLayer(0.4, 0.8)) if NOISE else None
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization()) if BATCH_NORM else None
    #model.add(NoiseLayer(0.25, 0.4)) if NOISE else None
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    #model.add(BatchNormalization()) if BATCH_NORM else None
    #model.add(NoiseLayer(0.5,0.5)) if NOISE else None
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model = base_model()
model.summary()
tensorboard = TensorBoard(log_dir='./cnn_graphs_FBN_remove_layer6BN', histogram_freq=epochs, write_graph=True, write_images=False)

model = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[tensorboard])

