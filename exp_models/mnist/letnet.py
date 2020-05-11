import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Lambda
from keras.callbacks import LearningRateScheduler, TensorBoard
import keras.backend as K
def build_model(inputshape, drop=False, droprate=0.2):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal',
                     input_shape=inputshape, name='l1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='l2'))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal',name='l3'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),name='l4'))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal',name='l5'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate)))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal',name='l6'))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal',name='l7'))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001

import utils.load_data as datama
from keras import callbacks
def train(dataset,epochs=50,**kwargs):
    epochs = epochs
    # load data
    if not 'data' in kwargs:
        (x_train, y_train), (x_test, y_test), (img_rows, img_cols, nb_classes) = datama.getData(dataset)
    else:
        ((x_train, y_train), (x_test, y_test), nb_classes) = kwargs['data']
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    # build network
    model = build_model(x_train.shape[1:])
    print(model.summary())

    # set callback

    change_lr = LearningRateScheduler(scheduler)
    if not 'logfile' in kwargs:
        csvlog = callbacks.CSVLogger("./log/lenet_" + dataset + ".log", separator=',', append=False)
    else:
        csvlog = callbacks.CSVLogger(kwargs['logfile'], separator=',', append=False)

    if not 'bestModelfile' in kwargs:
        checkPoint = callbacks.ModelCheckpoint("./model/lenet_" + dataset + ".h5", monitor="val_acc",
                                           save_best_only=True, verbose=1)
    else:
        checkPoint = callbacks.ModelCheckpoint(kwargs['bestModelfile'], monitor="val_acc",
                                               save_best_only=True, verbose=1)

    cbks = [change_lr,csvlog,checkPoint]

    # start train
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test),
              shuffle=True)
