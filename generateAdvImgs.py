import utils.load_data as datama
import utils.attacker as attacker
import utils.tools as helper
import logging
import numpy as np
import foolbox
from keras.utils import to_categorical
import argparse
import os

def compute_shuffle_idx(size):
    (x_train, y_train), (_, _), (_, _, num_class) = datama.getData('mnist')
    idx_mnist = np.arange(x_train.shape[0])
    np.random.shuffle(idx_mnist)

    (x_train, y_train), (_, _), (_, _, num_class) = datama.getData('cifar10')
    idx_cifar10 = np.arange(x_train.shape[0])
    np.random.shuffle(idx_cifar10)
    data = {'mnist':idx_mnist[:size], 'fashion_mnist':idx_mnist[:size], 'cifar10': idx_cifar10[:size]}
    if not os.path.isdir('./adv_data/'):
        os.mkdir('./adv_data/')
    np.save('./adv_data/data_index.npy', data)
    return idx_mnist[:size], idx_cifar10[:size]


def selectData(model, dataname, name,idx_true, test=False):
    print("{} ,{}".format(dataname, name))
    (x_train, y_train),(x_test, y_test), (_,_, num_class) = datama.getData(dataname)
    if name == 'mlp':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    selectedIndex = idx_true[dataname]
    (x_train, y_train) =  (x_train[selectedIndex], y_train[selectedIndex])
    assert name in weightspaths, "Error dict weightspath does not contain {}".format(name)

    #Test whether model weights is loaded correctly.
    scores = model.evaluate(x_train,y_train, verbose=0)
    assert scores[1]>0.85,"Do not load the trained-well weights. The model is not trained well."
    logger.info("Model {}, score {}".format(name, scores))

    del x_train, y_train
    y_test = np.squeeze(np.argmax(y_test, axis=1)[..., np.newaxis])
    return x_test, y_test,np.arange(x_test.shape[0])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m','--model', type=str,default='mlp')
    ap.add_argument('-d', '--dataset', type=str, default='mnist')
    ap.add_argument('-s', '--size', type=int, default=10000)
    ap.add_argument('-t', '--usetestdata', action='store_true')
    ap.add_argument('-a','--attacker', nargs='+')
    args = vars(ap.parse_args())

    size = args['size']
    dataname = args['dataset']
    attack_methods = args['attacker']
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler('./attack_logs/%s_%s_attack.log'%(args['model'], dataname))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Model %s, Dataset %s"%(args['model'], dataname))
    #model_cifar10 = ['dense_net', 'netinnet', 'resnet', 'vgg']
    path = './cluster_results/model/'
    cifarmodelweights = {'dense_net': 'densenet_cifar10.h5',
                    'netinnet':'NetInNet_cifar10.h5', 'resnet':'ResNet_cifar10.h5',
                    'vgg':'vgg_cifar10.h5'}
    mnistmodelweights = {'mnist':{'deepxplore':'deepxplore_mnist.hdf5',
                                  'lenet':'lenet_mnist.h5', 'mlp':'mlp_mnist.h5'},
                         'fashion_mnist':{'deepxplore':'deepxplore_fashion_mnist.hdf5',
                                          'lenet':'lenet_fashion_mnist.h5',
                                          'mlp':'mlp_fashion_mnist.h5'}}
    #model_mnist = ['mlp','deepxplore', 'lenet']
    datalist = ['mnist', 'fashion_mnist','cifar10']
    #modellist = {'mnist':model_mnist,'fashion_mnist':model_mnist,
    #             'cifar10':model_cifar10}

    if os.path.isfile('./adv_data/data_index.npy'):
        data_idx = np.load('./adv_data/data_index.npy', allow_pickle=True).item()
        idx_mnist, idx_cifar10 = data_idx.get('mnist'), data_idx.get('cifar10')
    else:
        print("warning, is creating a new index matrix")
        idx_mnist, idx_cifar10 = compute_shuffle_idx(size)
    idx_true = {'mnist':idx_mnist, 'fashion_mnist':idx_mnist, 'cifar10':idx_cifar10}
    if 'cifar' in dataname:
        # if you normalize data to 0 and 1
        #bounds = (0, 1.0)
        # if the image value range is still between 0 and 255.
        bounds = (0, 255.0)
        weightspaths = cifarmodelweights
    else:
        # if you normalize data to 0 and 1
        bounds = (0, 1.0)
        # if the image value range is still between 0 and 255.
        #bounds = (0, 255.0)
        weightspaths = mnistmodelweights[dataname]


    name = args['model']

    (x_train, _), (_, _), (_, _, num_class) = datama.getData(dataname)
    bestModelName = path + weightspaths[name]
    if name == 'mlp':
        x_train = x_train.reshape(x_train.shape[0], -1)
        # x_test = x_test.reshape(x_test.shape[0], -1)
    model = helper.load_model(name, bestModelName, x_train.shape[1:], num_class, isDrop=False)

    del x_train


    if not os.path.isfile('./adv_data/slectedTest10000ImgsIdx'):
            cx_train, cy_train, cidx = selectData(model, dataname, name, idx_true, test=True)
            ridx = np.arange(cidx.shape[0])
            np.random.shuffle(ridx)
            sidx = ridx[:10000]
            cx_train, cy_train, cidx = cx_train[sidx], cy_train[sidx], np.arange(10000)
            selectedIndex = cidx
            np.save('./adv_data/slectedTest1000ImgsIdx', {"idx":selectedIndex})
    else:
            data = np.load('./adv_data/slectedTest10000ImgsIdx').item()
            selectedIndex = data.get("idx")
            (_, _), (x_test, y_test), (_, _, num_class) = datama.getData(dataname)
            y_test = np.argmax(y_test, axis=1)[..., np.newaxis]
            cx_train, cy_train, cidx = x_test[selectedIndex], y_test[selectedIndex], np.arange(10000)


    if name == 'mlp':
        cx_train = cx_train.reshape(cx_train.shape[0], -1)

    if "FGSM" in attack_methods:
        print("FGSM Attacking")
        x_fgsm, s_fgsm = attacker.attackFGSM(model, cx_train, cy_train, bounds)
        idx_fgsm = cidx[s_fgsm]
        score = model.evaluate(x_fgsm, to_categorical(cy_train[s_fgsm]),verbose=0, batch_size=1)
        logger.info("FGSM {}, {}".format(name, score))
        data = {'fgsm': {'x_adv': x_fgsm, 'y_adv': cy_train[s_fgsm], 'idx': selectedIndex[idx_fgsm]}}
        np.save('./adv_data/%s_%s_fgsm.np' % (name, dataname), data)
        del x_fgsm,idx_fgsm,data

    if "DF" in attack_methods:
        print("DF Attacking")
        x_df, s_df = attacker.attackDeepFool(model, cx_train, cy_train, bounds)
        idx_df = cidx[s_df]
        score = model.evaluate(x_df, to_categorical(cy_train[s_df]),verbose=0, batch_size=1)
        logger.info("DF {}, {}".format(name, score))
        data = { 'df': {'x_adv': x_df, 'y_adv': cy_train[s_df], 'idx': selectedIndex[idx_df]}}
        np.save('./adv_data/%s_%s_df.np' % (name, dataname), data)
        del x_df,idx_df, data

    if "BIM" in attack_methods:
        print("BIM Attacking")
        x_bim, s_bim = attacker.attackBIM(model, cx_train, cy_train, bounds)
        idx_bim = cidx[s_bim]
        score = model.evaluate(x_bim, to_categorical(cy_train[s_bim]),verbose=0, batch_size=1)
        logger.info("BIM {}, {}".format(name, score))
        data = {'bim': {'x_adv': x_bim, 'y_adv': cy_train[s_bim], 'idx': selectedIndex[idx_bim]}}
        np.save('./adv_data/%s_%s_bim.np' % (name, dataname), data)
        del x_bim,s_bim,data

    if 'JSMA' in attack_methods:
        print("JSMA Attacking")
        x_jsma, s_jsma = attacker.attackJSMA(model, cx_train, cy_train, bounds)
        idx_jsma = cidx[s_jsma]
        score = model.evaluate(x_jsma, to_categorical(cy_train[s_jsma]),verbose=0, batch_size=1)
        logger.info("JSMA {}, {}".format(name, score))
        data = {'jsma': {'x_adv': x_jsma, 'y_adv': cy_train[s_jsma], 'idx': selectedIndex[idx_jsma]}}
        np.save('./adv_data/%s_%s_jsma.np'%(name, dataname), data)
        del x_jsma,idx_jsma,data

    if 'CW' in attack_methods:
        print("CW Attacking")
        x_cw, s_cw = attacker.attackCWl2(model, cx_train, cy_train, bounds)
        idx_cw = cidx[s_cw]
        score = model.evaluate(x_cw, to_categorical(cy_train[s_cw]), verbose=0, batch_size=1)
        logger.info("CW {}, {}".format(name, score))
        data = {'cw':{'x_adv':x_cw, 'y_adv':cy_train[s_cw], 'idx': selectedIndex[idx_cw]}}
        np.save('./adv_data/%s_%s_cw.np'%(name, dataname), data)



