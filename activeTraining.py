import numpy as np
import random as rn
# Numpy set random seed
np.random.seed(422)
# Python set random seed
rn.seed(42345)
import numpy as np
import utils.activateLearning as tools
import argparse
import utils.tools as helper
import os
import exp_models.cifar.Network_in_Network as netinnet
import exp_models.cifar.VGG16_Clipped as vgg
import exp_models.mnist.deepxplore as deepxplore
import exp_models.mnist.letnet as letnet
import exp_models.mnist.mlp as mlp
from keras.utils import to_categorical
from utils.ncoverage import NCoverage

def count_sampleNumOfEachClass(y, nb_class, num):
    '''
    The function is used to guarantee that the samples have the same class distribution with the original
    dataset.
    :param y: the ground truth labels of dataset
    :param nb_class: the number of classes in the data set
    :param num: the number of the selected samples
    :return: a dict that contains the number of samples for each class.
    '''
    dict = {}
    total = 0
    total_dict = {}
    for i in range(nb_class):
        if np.sum(y == i) <= num:
            dict[i] = np.sum(y == i)
            total += dict[i]
        else:
            dict[i] = num
            total += dict[i]
            total_dict[i] = np.sum(y == i)-num
    if total == num*nb_class:
        return dict
    else:
        leftk = list(total_dict.keys())
        lefv = np.sum(list(total_dict.values()))
        while(total != num*nb_class and lefv>0):
            id = np.random.randint(low=0, high=len(leftk), size=1)[0]
            if total_dict[leftk[id]]>0:
                dict[leftk[id]] += 1
                total += 1
                total_dict[leftk[id]] -= 1
            else:
                total_dict.pop(leftk[id], 'None')
            lefv = np.sum(list(total_dict.values()))
            leftk = list(total_dict.keys())
    return dict


# For SADL, which layers are used.
# You can modiy them as you want by changing the list of the layer name.
layersList ={"dense_net":['l293', 'l296'],
             "netinnet": ['l30', 'l27'],
             "resnet":['l92','l89'],
             "vgg":['dense_2', 'block3_conv3'],
             "deepxplore":['fc1','fc2'],'lenet':['l5','l6'], 'mlp':['dense_2']}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="mlp")
    ap.add_argument("-s", "--sort", type=str, default="var")
    ap.add_argument("-d", "--dataset", type=str, default="mnist")
    ap.add_argument("-e", "--epochs", type=int, default=300)
    ap.add_argument("-w", "--windows", type=int, default=5000)
    ap.add_argument("-r", "--droprate", type=float, default=0.3)
    ap.add_argument("-t", "--droptimes", type=int, default=50)

    args = vars(ap.parse_args())
    step = args['windows']
    epochs = args['epochs']
    drop_rate = args['droprate']
    drop_rep = args['droptimes']
    dataset = args['dataset']
    file = "./reTrainData/"+dataset+".npy"
    if not os.path.isdir('./log/'+args['model']):
        os.makedirs('./log/'+args['model'])
    if not os.path.isdir('./model/'+args['model']):
        os.makedirs('./model/'+args['model'])
    if not os.path.isdir('./reTrainData'):
        os.mkdir('./reTrainData')
    # control how many times of the repeation of the experiments
    exp_rep = 3
    for j in range(exp_rep):
        #load data
        if not os.path.isfile(file):
            assert "No Splitting Data. Run splitting_Training_dataset.py"
        else:
            data = np.load(file, allow_pickle = True ).item()
            (x_train, y_train), (x_remaining, y_remaining), (x_test, y_test) \
                = (data.get("x_train"), data.get("y_train")), \
                  (data.get("x_remaining"), data.get("y_remaining")), \
                  (data.get("x_test"), data.get("y_test"))
            num_class = y_train.shape[1]
            if args['model'] == "mlp":
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_test = x_test.reshape(x_test.shape[0], -1)
                x_remaining = x_remaining.reshape(x_remaining.shape[0], -1)


        set1 = len(x_train)
        set2 = len(x_remaining)
        set3 = len(x_test)
        num_step = set2 // step + 1
        isPrint = True
        assert os.path.isfile(file), 'Initial Data does not exist!'

        logsPath = "./log/%s/rep_%s_%s_%d" % (args['model'], args['dataset'], args['sort'],j)
        basemodelpath = "./model/%s/rep_%s_%s_%d" % (args['model'], args['dataset'], args['sort'],j)
        if not os.path.isdir(logsPath):
            try:
                os.mkdir(logsPath)
            except:
                print(logsPath+" was created.")

        if not os.path.isdir(basemodelpath):
            try:
                os.mkdir(basemodelpath)
            except:
                print(basemodelpath + " was created.")

        for i in np.arange(num_step):
            # the path that is used to store the model.
            bestModelName = os.path.join(basemodelpath, "%s_%s_%s_%d.h5"%
                                         (args['model'], args['dataset'],args['sort'],i))
            logfile = os.path.join(logsPath, "%s_%s_%s_%d.log"%
                                         (args['model'], args['dataset'],args['sort'],i))

            if args['model'] == 'netinnet':
                netinnet.train(dataset, epochs=epochs,data = ((x_train, y_train), (x_test, y_test),num_class),
                               logfile = logfile, bestModelfile=bestModelName)

            if args['model'] == "vgg":
                vgg.train_VGG16(dataset, num_epoch=epochs,data = ((x_train, y_train), (x_test, y_test),num_class),
                               logfile = logfile, bestModelfile=bestModelName)

            if args['model'] == "deepxplore":
                deepxplore.train(dataset, epochs=epochs,data = ((x_train, y_train), (x_test, y_test),num_class),
                               logfile = logfile, bestModelfile=bestModelName)

            if args['model'] == "lenet":
                letnet.train(dataset, epochs=epochs,data = ((x_train, y_train), (x_test, y_test),num_class),
                               logfile = logfile, bestModelfile=bestModelName)

            if args['model'] == "mlp":
                mlp.train(dataset, epochs=epochs,data = ((x_train, y_train), (x_test, y_test),num_class),
                               logfile = logfile, bestModelfile=bestModelName)

            model = helper.load_model(args['model'], bestModelName,
                                      x_train.shape[1:],num_class ,isDrop=False)

            # if there is not left data, break
            if len(y_remaining) == 0:
                break

            samples = (x_remaining, np.squeeze(helper.oneHot2Int(y_remaining)))
            ref_sample_num = step

            # if the score method is a kind of Neuron Coverage
            ncComputor = None
            if 'NC' in args['sort']:
                ncComputor = NCoverage(model, threshold=0.2)
                # there are two way to compute NC.
                # 1. only use the correctly predicted data
                # You can sample some data to compute NC
                # pl = np.argmax(model.predict(x_train), axis=1)
                # gl = np.argmax(y_train, axis=1)
                # inidata = x_train[pl==gl]

                # 2. use all the training dataset
                inidata = x_train[np.random.shuffle(np.arange(len(x_train)))]
                if 'KMNC' in args['sort'] or "SANC" in args['sort'] or "BNC" in args['sort'] or "Diff" in args['sort']:
                    ncComputor.initKMNCtable(inidata, './nc_data/%s_%s_%d_%d.p'%(args['model'],
                                                         args['dataset'], i,j), read=False, save=False)

            xref = None
            layers = None
            selectLayers = None
            if args["sort"]=="DSA" or args["sort"]=="LSA" or args["sort"]=="silhoutte":
                # get the selected layers. The output of these layers will be used to score the tests.
                ylabel = model.predict(x_train)
                y = np.argmax(ylabel, axis=1)
                print(y.shape)
                print(np.bincount(y))
                y_ref = np.squeeze(helper.oneHot2Int(y_train))
                random_ref_idx = np.arange(len(x_train))
                np.random.shuffle(random_ref_idx)
                xref = (x_train[random_ref_idx[:ref_sample_num]], y_ref[random_ref_idx[:ref_sample_num]], y[random_ref_idx[:ref_sample_num]]) #train_data, train_label, predict_label
                num_layers = len(model.layers)
                a = np.arange(num_layers)
                selectLayers =  layersList[args['model']]

            # enable Dropout layer
            model_drop = helper.load_model(args['model'], bestModelName,
                                      x_train.shape[1:],num_class ,isDrop=True, droprate=drop_rate)

            # compute each samples number for each label
            num_each = count_sampleNumOfEachClass(helper.oneHot2Int(y_remaining), num_class, step // num_class)
            # variance threhold for SADL
            varthreshold = {'mlp': {'mnist': 1e-2, 'fashion_mnist': 1e-1},
                            'deepxplore': {'mnist': 1e-5, 'fashion_mnist': 1e-5},
                            'lenet': {'mnist': 1e-5, 'fashion_mnist': 1e-5},
                            'vgg': {'cifar10': 1e-1}, 'netinnet': {'cifar10': 1e-1},
                            }
            # Select Data from the remaining set
            (X_jointrain, y_jointrain), (x_remaining, y_remaining), _ = \
                tools.getSamples(model_drop if args["sort"] in ["var", "KL", "Hist", "varP", "KLP", "varW"]  else model,
                                 samples, step, drop_rep=drop_rep,
                                 method=args["sort"], num_class=num_class,
                                 xref=xref, layers=selectLayers,
                                 ref_sample_num=ref_sample_num,
                                 num_each=num_each,
                                 orginalModel=model, ncComputor=ncComputor,
                                 varthreshold=varthreshold[args['model']][dataset])
            if len(y_jointrain) != 0:
                y_jointrain = to_categorical(y_jointrain, num_class)
            if len(y_remaining) != 0:
                y_remaining = to_categorical(y_remaining, num_class)

            if len(X_jointrain)!=0:
                x_train = np.concatenate((x_train, X_jointrain))
                y_train = np.concatenate((y_train, y_jointrain))

            del model
            del model_drop
            del ncComputor







