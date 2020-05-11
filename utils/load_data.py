from  keras.datasets import cifar10,mnist,cifar100,fashion_mnist
from keras.utils import to_categorical
import numpy as np
def getData(name="cifar10"):
    if name=="cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # normalize data, it can help the model learn faster.
        # But it is not necessary except the value range is very large.
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        img_rows, img_cols = x_train.shape[1],x_train.shape[2]
        return (x_train, y_train), (x_test, y_test),(img_rows, img_cols, 10)

    if name=="cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # normalize data, it can help the model learn faster.
        # But it is not necessary except the value range is very large.
        # x_train = x_train.astype('float32')/255
        # x_test = x_test.astype('float32')/255
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]
        return (x_train, y_train), (x_test, y_test),(img_rows, img_cols, 100)

    if name=="mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        x_train /= 255.
        x_test /= 255.
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]
        return (x_train, y_train), (x_test, y_test),(img_rows, img_cols, 10)

    if name=="fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        x_train /= 255.
        x_test /= 255.
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]
        return (x_train, y_train), (x_test, y_test),(img_rows, img_cols, 10)
    raise Exception("No the dataset: "+name+".")


def initializeData(x_train, y_train, trainsize):
     '''

     :param X_train:
     :param y_train:
     :param X_test:
     :param y_test:
     :param trainsize:
     :param sesize2:
     :param testsize:
     :return:
     '''

     idx = np.arange(x_train.shape[0])
     np.random.shuffle(idx)
     idx_train = idx[:trainsize]
     idx_remaining = idx[trainsize:]
     return (x_train[idx_train], y_train[idx_train]), (x_train[idx_remaining], y_train[idx_remaining])

reTrainDataList = {"mnist":"./reTrainData/mnist.npy",
                   "fashion_mnist":"./reTrainData/fashion_mnist.npy",
                   "cifar10":"./reTrainData/cifar10.npy"}
import os
def split_data(training_size):
    if os.path.isdir("./reTrainData"):
        os.makedirs("./reTrainData", exist_ok=True)
    set = ['mnist', 'fashion_mnist', 'cifar10']
    for dataset in set:
        (x_train, y_train), (x_test, y_test), (_, _, num_class) = getData(dataset)
        (x_train, y_train), (x_remaining, y_remaining) = initializeData(x_train, y_train, training_size)
        data = {"x_train": x_train, "y_train": y_train,
                "x_test": x_test, "y_test": y_test,
                "x_remaining": x_remaining, "y_remaining": y_remaining}
        np.save(reTrainDataList[dataset], data)

def spli_toy_data(step=300):
    set = ['mnist', 'fashion_mnist', 'cifar10']
    for dataset in set:
        (x_train, y_train), (x_test, y_test), (_, _, num_class) = getData(dataset)
        (x_train, y_train), (x_test, y_test) = (x_train[:500], y_train[:500]), (x_test[:500], y_test[:500])
        (x_train, y_train), (x_remaining, y_remaining) = initializeData(x_train, y_train, step)
        data = {"x_train": x_train, "y_train": y_train,
                "x_test": x_test, "y_test": y_test,
                "x_remaining": x_remaining, "y_remaining": y_remaining}
        np.save(reTrainDataList[dataset], data)

