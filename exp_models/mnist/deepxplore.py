import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Convolution2D, MaxPooling2D,Dropout,Dense,Activation,Flatten
from keras import Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import utils
import os
'''
The model is from Paper DeepXplore open source code.
'''
def Model3_deepXplore(input_shape=(28,28, 1),
                     drop_rate=0.4, nb_classes=10, kernel_size=(5,5), drop=False):
    # block1
    model = Sequential()
    model.add(Convolution2D(32, kernel_size, activation='relu',
                            padding='same', name='block1_conv1', input_shape=input_shape)) #1
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1')) #2
    model.add(Dropout(drop_rate))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, drop_rate)))
    # block2
    model.add(Convolution2D(64, kernel_size, activation='relu', padding='same', name='block2_conv1'))#4
    model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))#5
    model.add(Dropout(drop_rate))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, drop_rate)))
    model.add(Flatten(name='flatten'))

    model.add(Dense(120, activation='relu', name='fc1'))# -5
    model.add(Dropout(drop_rate))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, drop_rate)))
    model.add(Dense(84, activation='relu', name='fc2'))# -3
    model.add(Dense(nb_classes, name='before_softmax'))# -2
    model.add(Activation('softmax', name='predictions'))#
    return model


import utils.load_data as datama
def train(dataset, epochs = 50, drop_rate=0.4,**kwargs):
    # input image dimensions
    epochs = epochs
    if not 'data' in kwargs:
        (x_train, y_train), (x_test, y_test), (img_rows, img_cols, nb_classes) = datama.getData(dataset)
    else:
        ((x_train, y_train), (x_test, y_test), nb_classes) = kwargs['data']
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    input_shape = (img_rows, img_cols, 1)

    print("X train shape {}".format(x_train.shape))
    print("y train shape {}".format(y_train.shape))
    print("X test shape {}".format(x_test.shape))
    print("y test shape {}".format(y_test.shape))

    model = Model3_deepXplore(input_shape=input_shape, drop_rate=drop_rate, nb_classes=nb_classes)
    model.summary()
    if not 'bestModelfile' in kwargs:
        bestModel = "./model/deepxplore_"+dataset+".hdf5"
    else:
        bestModel = kwargs['bestModelfile']

    if not 'logfile' in kwargs:
        log =  "./log/deepxplore_"+dataset+".log"
    else:
        log = kwargs['logfile']
    checkpointer = ModelCheckpoint(filepath=bestModel, verbose=1, save_best_only=True,
                                   monitor="val_acc")
    logger = CSVLogger(log, separator=",", append=False)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[checkpointer, logger])



