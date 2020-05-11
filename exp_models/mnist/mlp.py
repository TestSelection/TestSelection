'''
Trains a simple deep NN on the MNIST dataset.
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Lambda
from keras.optimizers import RMSprop
import keras.backend as K

def mlp(num_classes,inputshape,drop=False, droprate=0.2):
    model = Sequential()
    #print(inputshape)
    model.add(Dense(512, activation='relu', input_shape=inputshape, name='dense_1'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate), name='lambda1'))
    else:
        model.add(Dropout(0.2, name='d1'))
    model.add(Dense(512, activation='relu', name='dense_2'))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=droprate), name='lambda2'))
    else:
        model.add(Dropout(0.2, name='d2'))
    model.add(Dense(num_classes, activation='softmax', name='dense_3'))
    #model.summary()
    return model

import utils.load_data as datama
from keras import  callbacks
def train(dataset, epochs=50,**kwargs):
    batch_size = 128
    epochs = epochs
    if not 'data' in kwargs:
        (x_train, y_train), (x_test, y_test), (img_rows, img_cols, num_classes) = datama.getData(dataset)
    else:
        ((x_train, y_train), (x_test, y_test), num_classes) = kwargs['data']
        #img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    model = mlp(num_classes, inputshape=x_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    if not 'logfile' in kwargs:
        csvlog = callbacks.CSVLogger("./log/mlp_" + dataset + ".log", separator=',', append=False)
    else:
        csvlog = callbacks.CSVLogger(kwargs['logfile'], separator=',', append=False)

    if not 'bestModelfile' in kwargs:
        checkPoint = callbacks.ModelCheckpoint("./model/mlp_" + dataset + ".h5", monitor="val_acc",
                                           save_best_only=True, verbose=1)
    else:
        checkPoint = callbacks.ModelCheckpoint(kwargs['bestModelfile'], monitor="val_acc",
                                               save_best_only=True, verbose=1)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[csvlog,checkPoint])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
