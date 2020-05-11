import numpy as np
np.random.seed(698686)
print("Set Random Seed 698686")
from keras.layers import Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras import backend as K
from keras.layers.core import Lambda
from keras.datasets import cifar10
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import optimizers
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
import utils.load_data as datama
def train_VGG16(dataset, num_epoch=300, batchSize=128, **kwargs):
    if not 'data' in kwargs:
        (X_train, Y_train), (X_test,Y_test), (img_rows, img_cols, nb_class) = datama.getData(dataset)
    else:
        ((X_train, Y_train), (X_test, Y_test), nb_class) = kwargs['data']
        img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    if not 'bestModelfile' in kwargs:
        bestmodelname = "./model/vgg_"+dataset+".h5"
    else:
        bestmodelname = kwargs['bestModelfile']

    if not 'logfile' in kwargs:
        logpath = "./log/vgg_"+dataset+".log"
    else:
        logpath = kwargs['logfile']

    # VGG16 Original Weights
    weights_path = './model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    # if not regularization:
    print("VGG16_clipped")
    model = VGG16_clipped(input_shape=X_train.shape[1:], rate=0.4,
                          nb_classes=nb_class)  # VGG16_clipped(input_shape=(32,32,3), rate=0.2, nb_classes=10, drop=False)
    vgg16 = VGG16(weights='imagenet', include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])
    for l in model.layers:
        if l.name in layer_dict:
            model.get_layer(name=l.name).set_weights(layer_dict[l.name].get_weights())
    #model.load_weights(weights_path, by_name=True)
    checkPoint = ModelCheckpoint(bestmodelname, monitor="val_acc", save_best_only=True, verbose=1)
    model.summary()
    num_layers = len(model.layers)
    a = np.arange(num_layers)
    layers = a[(num_layers - 6):-2]
    print(layers)
    lr = 1e-2

    def lr_scheduler(epoch):
        initial_lrate = lr
        drop = 0.9
        epochs_drop = 50.0
        lrate = initial_lrate * np.power(drop,
                                         np.floor((1 + epoch) / epochs_drop))
        return lrate

    reduce_lr = callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    # compile the model with SGD/momentum optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr, momentum=0.9),
                  metrics=['accuracy'])
    csvlog = callbacks.CSVLogger(logpath, separator=',', append=False)

    # data augmentation
    # if you do not want to use data augmentation, comment below codes.
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    train_datagen.fit(X_train)
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=batchSize)

    # fine-tune the model
    nb_train_samples = X_train.shape[0] // batchSize
    nb_epoch = num_epoch
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        callbacks=[checkPoint, reduce_lr, csvlog])


    # model.fit(X_train, Y_train,
    #           batch_size=128,
    #           epochs=num_epoch,
    #           callbacks=[checkPoint, reduce_lr, csvlog],
    #           validation_data=(X_test, Y_test),
    #           shuffle=True)
    del model


def VGG16_clipped(input_shape=None, rate=0.2, nb_classes=10, drop=False):
    # Block 1
    model = Sequential()
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1', input_shape=input_shape)) #1
    model.add(BatchNormalization(name="batch_normalization_1"))     #2
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')) #3
    model.add(BatchNormalization(name="batch_normalization_2"))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))#4

    # Block 2
    model.add(Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1'))#5
    model.add(BatchNormalization(name="batch_normalization_3"))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2'))#6
    model.add(BatchNormalization(name="batch_normalization_4"))#7
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))#8

    # Block 3
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1'))#9
    model.add(BatchNormalization(name="batch_normalization_5")) #10
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2'))#11
    model.add(BatchNormalization(name="batch_normalization_6"))

    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3'))#12
    model.add(BatchNormalization(name="batch_normalization_7")) #13
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')) #14
    model.add(Flatten())#15
    model.add(Dense(256, activation='relu', name='dense_1')) #16
    model.add(BatchNormalization(name="batch_normalization_8"))#17
    model.add(Dense(256, activation='relu', name='dense_2'))#18
    model.add(BatchNormalization(name="batch_normalization_9"))#19
    model.add(Dense(nb_classes, activation='softmax', name='dense_3')) #20
    return model

# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input, decode_predictions
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     input_img_data = image.img_to_array(img)
#     input_img_data = np.expand_dims(input_img_data, axis=0)
#     input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
#     return input_img_data
