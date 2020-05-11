'''
Network In Network, https://arxiv.org/pdf/1312.4400.pdf
'''
import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras import backend as K
import utils.load_data as datama



def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test

def scheduler_bn(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004

def scheduler_nonBn(epoch):
    if epoch <= 80:
        return 0.01
    if epoch <= 140:
        return 0.005
    return 0.001


def build_model_bn(intpu_shape,dropout,weight_decay, drop=False,droprate=0.2):
    model = Sequential()
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     input_shape=intpu_shape, name='l1'))#x_train.shape[1:]))
    model.add(BatchNormalization( name='l2'))
    model.add(Activation('relu',name='l3'))
    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l4'))
    model.add(BatchNormalization(name='l5'))
    model.add(Activation('relu',name='l6'))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l7'))
    model.add(BatchNormalization(name='l8'))
    model.add(Activation('relu',name='l9'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',name='l10'))

    model.add(Dropout(dropout,name='l11'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),name='l12'))
    model.add(BatchNormalization(name='l13'))
    model.add(Activation('relu',name='l14'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),name='l15'))
    model.add(BatchNormalization(name='l16'))
    model.add(Activation('relu',name='l17'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),name='l18'))
    model.add(BatchNormalization(name='l20'))
    model.add(Activation('relu',name='l21'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',name='l22'))

    model.add(Dropout(dropout,name='l23'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),name='l24'))
    model.add(BatchNormalization(name='l25'))
    model.add(Activation('relu',name='l26'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),name='l27'))
    model.add(BatchNormalization(name='l28'))
    model.add(Activation('relu',name='l29'))
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),name='l30'))
    model.add(BatchNormalization(name='l31'))
    model.add(Activation('relu',name='l32'))

    model.add(GlobalAveragePooling2D(name='l33'))
    model.add(Activation('softmax',name='l34'))

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def build_model_nonBn(input_shape,dropout,weight_decay,drop=False,droprate=0.2):
    model = Sequential()
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     input_shape=input_shape, name='l1'))  # x_train.shape[1:]))
    model.add(Activation('relu', name='l3'))
    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l4'))
    model.add(Activation('relu', name='l6'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l7'))
    model.add(Activation('relu', name='l9'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='l10'))
    model.add(Dropout(dropout, name='l11'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l12'))
    model.add(Activation('relu', name='l14'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l15'))
    model.add(Activation('relu', name='l17'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l18'))

    model.add(Activation('relu', name='l21'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='l22'))
    model.add(Dropout(dropout, name='l23'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l24'))
    model.add(Activation('relu', name='l26'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l27'))
    model.add(Activation('relu', name='l29'))
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), name='l30'))
    model.add(Activation('relu', name='l32'))

    model.add(GlobalAveragePooling2D(name='l33'))
    model.add(Activation('softmax', name='l34'))

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

from  keras import callbacks
#batch_size    = 128
#epochs        = 200
#num_classes   = 10
dropout       = 0.5
weight_decay  = 0.0001
#log_filepath  = './nin_bn'

def train(dataset='cifar10', name="bn",**kwargs):
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 128
    epochs =  kwargs['epochs'] if 'epochs' in kwargs else 300
    dropout = kwargs['dropout']  if 'dropout' in kwargs else 0.5
    weight_decay = kwargs['weight_decay']  if 'weight_decay' in kwargs else 0.0001
    log_filepath = './'+name
    models = {"bn":build_model_bn, "nonBn":build_model_nonBn}
    schedulers = {"bn":scheduler_bn, "nonBn":scheduler_nonBn}
    # load data
    if not 'data' in kwargs:
        (x_train, y_train), (x_test, y_test), (img_rows, img_cols, num_classes) = datama.getData(dataset)
    else:
        ((x_train, y_train), (x_test, y_test), num_classes) = kwargs['data']
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]

    # build network
    model = models[name](x_train.shape[1:],dropout,weight_decay)
    print(model.summary())

    change_lr = LearningRateScheduler(schedulers[name])
    if not 'logfile' in kwargs:
        csvlog = callbacks.CSVLogger("./log/netinnet_" + dataset + ".log", separator=',', append=False)
    else:
        csvlog = callbacks.CSVLogger(kwargs['logfile'], separator=',', append=False)

    if not 'bestModelfile' in kwargs:
        checkPoint = callbacks.ModelCheckpoint('./model/netinnet_' + dataset + ".h5", save_best_only=True, monitor="val_acc",  verbose=1)
    else:
        checkPoint = callbacks.ModelCheckpoint(kwargs['bestModelfile'], monitor="val_acc",
                                           save_best_only=True, verbose=1)
    cbks = [change_lr,csvlog, checkPoint]
    # set data augmentation
    # if you do not want to use data augmentation, comment below codes.
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)
    iterations = x_train.shape[0]//batch_size
    #start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                        epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))

    # start train
    # train without data augmentation
    # model.fit(x_train, y_train,
    #           batch_size=128,
    #           epochs=epochs,
    #           callbacks=cbks,
    #           validation_data=(x_test, y_test),
    #           shuffle=True)

