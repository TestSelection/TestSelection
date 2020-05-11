from keras.models import load_model
#import foolbox
import numpy as np
#from tqdm import tqdm
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
import exp_models.cifar.Network_in_Network as netinnet
import exp_models.cifar.VGG16_Clipped as vgg
import exp_models.mnist.deepxplore as deepxplore
import exp_models.mnist.letnet as letnet
import exp_models.mnist.mlp as mlp
from keras import Model
from keras import optimizers, regularizers


def get_coverage(low, uppper,k,  score):
    """Surprise Coverage
    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        score (list): List of score
    Returns:
        cov (int): Surprise coverage.
    """
    #vmax, vmin = np.max(score), np.min(score)
    #normalized = (score - vmin)/(vmax - vmin)

    buckets = np.digitize(score, np.linspace(low, uppper, k))
    return len(list(set(buckets))) / float(k) * 100

def get_Silhouettecoverage(k, score):
    """Surprise Coverage
    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        score (list): List of score
    Returns:
        cov (int): Surprise coverage.
    """

    buckets = np.digitize(score, np.linspace(-1, 1, k))
    return len(list(set(buckets))) / float(k) * 100

def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def computeCor(x,y):

    x = x[np.argwhere(np.logical_not( np.isnan(y)))]
    y = y[np.argwhere(np.logical_not( np.isnan(y)))]
    x = np.squeeze(x)
    y = np.squeeze(y)
    (kendallscore,p) = kendalltau(x, y)
    cor = 0#distcorr(x, y)
    pcor = 0# np.corrcoef(x, y)[0][1]
    return np.round(kendallscore, 4), np.round(cor,4), np.round(pcor,4), p

# def generateAdversialExample(model, images, label, attackMethod, bounds=(0,255.)):
#     fmodel = foolbox.models.KerasModel(model, bounds=bounds)
#     attack = attackMethod(fmodel)
#     adversial = np.zeros(images.shape, dtype=float)
#
#     for i in np.arange(len(images)):
#          print("Generate adversial Image {}".format(i))
#          adversial[i] = attack(images[i], label[i][0])
#          if np.isnan(adversial).any():
#              where_are_NaNs = np.isnan(adversial[i])
#              adversial[i][where_are_NaNs] = images[i][where_are_NaNs]
#          pre_y = np.argmax(fmodel.predictions(adversial[i]))
#          print("Adv Img pre {}, {}".format(pre_y, label[i][0]))
#          #assert label[i][0]!=pre_y,"The adversial image has the same label with the ground truth. {}, {}".format(pre_y, label[i][0])
#     return adversial

def generateNoiseAdversialExample(images, mu, var, ):
    for i in np.arange(len(images)):
        print("Generate adversial Image {}".format(i))
        images[i] = images[i] + np.random.normal(mu[i], np.sqrt(var[i]), images[i].shape)
    return images


def predict(X, Y, num_repeat, num_class, model, F = None):
    '''

    :param X: input data
    :param Y: ground truth
    :param num_repeat: the number of repeating prediction
    :param num_class: the number of classes
    :param model: deep learning model
    :return: (output of the model, predicted label, counter for the correct times,
                the probability for the correct label, variance)
    '''
    result = np.zeros((num_repeat, X.shape[0], num_class))
    #print("resutl shape: {}".format(result.shape))
    label = np.zeros((num_repeat, X.shape[0]))
    #print("label shape: {}".format(label.shape))
    counter = np.zeros((num_repeat, X.shape[0]))
    #print("counter shape: {}".format(counter.shape))
    p_rlabel = np.zeros((num_repeat, X.shape[0]))
    #print("p_rlabel shape: {}".format(p_rlabel.shape))
    #from tqdm import tqdm
    for i in range(num_repeat):
        #print("Iteration {}".format(i))
        result[i] = model.predict(X, verbose=0) if F==None else F((X, 1))[0]
        #print(result[i][0][0])
        #print("result[i] shape: {}".format(result[i].shape))
        label[i] = np.argmax(result[i], axis=1)
        #print(result[i])
        #print("label[i] shape: {}".format(label[i].shape))
        Y = np.squeeze(Y)
        counter[i][Y == label[i]] += 1
        #print("counter[i] shape: {}".format(counter[i].shape))
        p_rlabel[i] = result[i][np.arange(X.shape[0]), Y]
        #print("p_rlabel[i] shape: {}".format(p_rlabel[i].shape))

    variance = np.var(p_rlabel, axis=0)
    means = np.mean(p_rlabel, axis=0)
    return (result, label, counter, p_rlabel, variance, means)

def prob_mean(result_test):
    mean_all_class = np.mean(result_test, axis=0)
    p_index = np.argmax(mean_all_class, axis=1)
    p = mean_all_class[np.arange(len(mean_all_class)), p_index]
    var = np.var(mean_all_class, axis=1)
    return p,var

def var_mean(result_test):
    var_all_class = np.var(result_test, axis=0)
    var_mean_all_class = np.mean(var_all_class, axis=1)
    return var_mean_all_class

def var_prediction(result, y_pre):
    var_all_class = np.var(result, axis=0)
    var_mean_all_class = var_mean(result)
    a = np.arange(len(var_mean_all_class))
    var_pre = var_all_class[a, np.squeeze(y_pre)]
    return var_pre

def loadTest_cifar10(file="./data/test_cifar10.npy"):
    data = np.load(file).item()
    (result_test, label_test, counter_test, p_test, var_test, means_test) = \
        data.get("result_test"), data.get("label_test"), data.get("counter_test"), data.get("p_test"), data.get(
            "var_test"), data.get("means_test")
    return (result_test, label_test, counter_test, p_test, var_test, means_test)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def getLayerIndex(layers, name):
    for i, n in enumerate(layers):
        if n.name == name:
            return i
    return -1

def setWeights(model, weights_path):
    import tensorflow as tf
    m = load_model(weights_path, custom_objects={"tf": tf})
    layer_dict = dict([(layer.name, layer) for layer in m.layers])
    for l in model.layers:
        if l.name in layer_dict:
            print(l.name)
            model.get_layer(name=l.name).set_weights(layer_dict[l.name].get_weights())
    return model



def computeKL(result,nb_classes,  label):
    nb_samples = result.shape[1]
    hist = np.zeros((nb_samples, nb_classes))
    for i in range(nb_samples):
        l_i = label[:, i]
        for j in range(nb_classes):
            #print(np.sum(l_i == j))
            hist[i][j] = np.sum(l_i == j)

    var_hist = np.var(hist/ label.shape[0], axis=1)

    hist_pdf = hist / label.shape[0]
    uniform_pdf = np.repeat(1 / nb_classes, nb_classes)
    kl = np.zeros(nb_samples)
    import scipy.stats
    for j in range(len(kl)):
        kl[j] = scipy.stats.entropy(np.squeeze(hist_pdf[j]), uniform_pdf)
    return kl,var_hist

from keras.layers import Input
import os
def load_model(model_name, weight_path, inputshape, num_classes,isDrop=False, droprate=0.3):
    assert os.path.isfile(weight_path), "No the weights {}".format(weight_path)


    if model_name == 'netinnet':
        model = netinnet.build_model_bn(inputshape, netinnet.dropout, netinnet.weight_decay,
                                        drop=isDrop, droprate=netinnet.dropout)
        model.load_weights(weight_path, by_name=True)
        return model


    if model_name == "vgg":
        model = vgg.VGG16_clipped(inputshape, drop=isDrop, rate=droprate, nb_classes=num_classes)
        model.load_weights(weight_path, by_name=True)
        # compile the model with SGD/momentum optimizer
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=.1, momentum=0.9),
                      metrics=['accuracy'])
        return model

    if model_name == "deepxplore":
       model = deepxplore.Model3_deepXplore(inputshape, drop_rate=droprate,nb_classes=num_classes,drop=isDrop)
       # compile
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       model.load_weights(weight_path, by_name=True)
       return model

    if model_name == "lenet":
       model = letnet.build_model(inputshape, drop=isDrop, droprate=droprate)
       # compile
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       model.load_weights(weight_path, by_name=True)
       return model

    if model_name == "mlp":
       model = mlp.mlp(num_classes, inputshape, drop=isDrop, droprate=droprate)
       # compile
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       model.load_weights(weight_path, by_name=True)
       return model

    if model_name == 'resnet50':
        from exp_models.imagenet.resnet50 import ResNet50
        # input image dimensions
        img_rows, img_cols = 224, 224
        input_shape = (img_rows, img_cols, 3)
        # define input tensor as a placeholder
        input_tensor = Input(shape=input_shape)
        resnet = ResNet50(input_tensor=input_tensor, isDrop=isDrop, drop_rate=droprate)
        return resnet

    if model_name == 'VGG16':
        from exp_models.imagenet.vgg16 import  VGG16
        # input image dimensions
        img_rows, img_cols = 224, 224
        input_shape = (img_rows, img_cols, 3)
        input_tensor = Input(shape=input_shape)
        vgg16 = VGG16(input_tensor=input_tensor, isDrop=isDrop, drop_rate=droprate)
        return vgg16

    if model_name == 'VGG19':
        from exp_models.imagenet.vgg19 import VGG19
        # input image dimensions
        img_rows, img_cols = 224, 224
        input_shape = (img_rows, img_cols, 3)
        input_tensor = Input(shape=input_shape)
        vgg19 = VGG19(input_tensor=input_tensor, isDrop=isDrop, drop_rate=droprate)
        return vgg19

    if model_name == 'inceptionv3':
        from exp_models.imagenet.inception_v3 import InceptionV3
        img_cols, img_rows = 299, 299
        input_shape = (img_rows, img_cols, 3)
        input_tensor = Input(shape=input_shape)
        netv3 = InceptionV3(input_tensor=input_tensor, isDrop=isDrop, drop_rate=droprate)
        return netv3





    assert False,"Not Defined the model name, {}".format(model_name)


def oneHot2Int(y):
    return np.argmax(y, axis=1)[..., np.newaxis]

def getSADLScore_Cifar10():
    file = "./data/sadl.npy"
    data = np.load(file).item()
    kdescores, dslscores = data.get("kdescores"), data.get("dslscores")
    return kdescores,dslscores

def getSADLSocre_MNISTFashion():
    file = "./data/mnist_fasion_sadl.npy"
    data = np.load(file).item()
    kdescores, dslscores = data.get("kdescores"), data.get("dslscores")
    return kdescores, dslscores

def getKLandHistVarScore_Cifar10():
    file = "./data/test_cifar10.npy"
    data = np.load(file).item()
    (result_test, label_test, counter_test, p_test, var_gt, means_test) = \
        data.get("result_test"), data.get("label_test"), data.get("counter_test"), data.get("p_test"), data.get(
            "var_test"), data.get("means_test")
    kl, var_hist = computeKL(result_test, 10, label_test)
    return kl, var_hist

def getKLandHistVarScore_MNIST():
    data = np.load("./data/mnist_fasion.npy")
    result, label, counter, p_rlabel, variance, means = data[0], data[1], data[2], data[3], data[4], data[5]
    kl, var_hist = computeKL(result, 10, label)
    return kl, var_hist

def getADVimagesScore_Cifar10():
    data = np.load("./data/cifar_adv_score.npy")
    dsascore = data.get('dsascore')
    lsascore = data.get('lsascore')
    varscore = data.get('varscore')
    klscore = data.get('KL')
    histscore = data.get('VarHist')
    names = ["fgsm", "jsma", "df", "X_join"]
    return dsascore, lsascore, varscore, klscore, histscore,names

def getADVimageScore_MNIST():
    data = np.load("./data/mnist_adv_score.npy")
    names = ["fgsm", "jsma", "df", "X_join"]
    dsascore = data.get('dsascore')
    lsascore = data.get('lsascore')
    varscore = data.get('varscore')
    klscore = data.get('KL')
    histscore = data.get('VarHist')
    return dsascore, lsascore, varscore, klscore, histscore.names

def getADVimages_MNIST():
    data_adv = np.load("./data/adv_mnist.npy").item()
    names = ["fgsm", "jsma", "df", "X_join"]
    return data_adv,names

def getADVimgs_Cifar():
    names = ["fgsm", "jsma", "df", "X_join"]
    data_adv = np.load("./data/cifar10_adv_5000.npy").item()
    return data_adv,names

def getMNIST_VoteLabel():
    data = np.load("./data/mnist_fasion.npy")
    result, label, counter, p_rlabel, variance, means = data[0], data[1], data[2], data[3], data[4], data[5]
    label = label.astype(int)
    votelabels = np.array([np.argmax(np.bincount(label[:, i])) for i in range(label.shape[1])])
    return votelabels

def getCifar_VoteLabel():
    file = "./data/test_cifar10.npy"
    data = np.load(file).item()
    (result_test, label_test, counter_test, p_test, var_gt, means_test) = \
        data.get("result_test"), data.get("label_test"), data.get("counter_test"), data.get("p_test"), data.get(
            "var_test"), data.get("means_test")
    votelabels = np.array([np.argmax(np.bincount(label_test[:, i])) for i in range(label_test.shape[1])])
    return votelabels

def countGroup(bins, vmin, values):
    width =  bins[1]-bins[0]
    idx_bins = np.minimum(np.floor((values-vmin)/width), len(bins)-1-1).astype(int)
    groups={}
    vgroups={}
    for i in range(len(bins)-1):
        groups[i] = np.nonzero(idx_bins==i)[0]
        vgroups[i] = values[idx_bins==i]
    return groups,vgroups

def getGroups_cifar10(nb_group=10, name="var"):
    file = "./data/test_cifar10.npy"
    data = np.load(file).item()
    (result_test, label_test, counter_test, p_test, var_gt, means_test) = \
        data.get("result_test"), data.get("label_test"), data.get("counter_test"), data.get("p_test"), data.get(
            "var_test"), data.get("means_test")
    if name=="var":
        values = var_mean(result_test)
    elif name=="p":
        values,_ = prob_mean(result_test)
    else:
        _, values =prob_mean(result_test)

    vmin, vmax = np.min(values), np.max(values)
    bins = np.linspace(vmin, vmax, nb_group)
    groups,vgroups = countGroup(bins, vmin, values)
    return groups,vgroups,bins

def getGroups_mnist(nb_group=10, name="var"):
    data = np.load("./data/mnist_fasion.npy")
    result, label, counter, p_rlabel, variance, means = data[0], data[1], data[2], data[3], data[4], data[5]
    if name=="var":
        values = var_mean(result)
    elif name=="p":
        values,_ = prob_mean(result)
    else:
        _, values = prob_mean(result)

    vmin, vmax = np.min(values), np.max(values)
    bins = np.linspace(vmin, vmax, nb_group+1)
    groups,vgroups = countGroup(bins, vmin, values)
    return groups,vgroups,bins,values



def getSurface_cifar(nb_group=10, name='var'):
    data = np.load("./data/test_cifar10.npy").item()
    result, label, counter, p_rlabel, variance, means = \
        data.get("result_test"), data.get("label_test"), data.get("counter_test"), data.get("p_test"), data.get(
            "var_test"), data.get("means_test")
    if name=='var':
        var = var_mean(result)
    elif name=="KL":
        var, _ = getKLandHistVarScore_Cifar10()
    elif name=="LSA":
        var,_ = getSADLScore_Cifar10()
    else:
        _, var = getSADLScore_Cifar10()
    p, _ = prob_mean(result)

    vmin, vmax = np.min(var), np.max(var)
    pmin, pmax = np.min(p), np.max(p)
    vbins = np.linspace(vmin, vmax, nb_group+1)
    pbins = np.linspace(pmin, pmax, nb_group+1)
    vwidth = vbins[1] - vbins[0]
    vidx_bins = np.minimum(np.floor((var - vmin) / vwidth), len(vbins) - 1 - 1).astype(int)

    pw = pbins[1] - pbins[0]
    pidx_bins =  np.minimum(np.floor((p - pmin) / pw), len(pbins) - 1 - 1).astype(int)

    dic = {}
    t = 0
    for i in range(nb_group):
        dic[i] = {}
        for j in range(nb_group):
            dic[i][j] = np.nonzero(np.logical_and(vidx_bins == i, pidx_bins==j))[0]
            t+=len(dic[i][j])

    return dic, var, p, vbins, pbins


def compute2DHistGroup(score, p, nb_group):
    vmin, vmax = np.min(score), np.max(score)
    pmin, pmax = np.min(p), np.max(p)
    vbins = np.linspace(vmin, vmax, nb_group + 1)
    pbins = np.linspace(pmin, pmax, nb_group + 1)
    vwidth = vbins[1] - vbins[0]
    if vwidth == 0.:
        vwidth = vwidth+0.0000001
    vidx_bins = np.minimum(np.floor((score - vmin) / vwidth), len(vbins) - 1 - 1).astype(int)

    pw = pbins[1] - pbins[0]
    if pw == 0.:
        pw = pw + 0.0000001
    pidx_bins = np.minimum(np.floor((p - pmin) / pw), len(pbins) - 1 - 1).astype(int)

    dic = {}
    t = 0
    for i in range(nb_group):
        dic[i] = {}
        for j in range(nb_group):
            a = np.nonzero(np.logical_and(vidx_bins == i, pidx_bins == j))[0]
            dic[i][j] = np.nonzero(np.logical_and(vidx_bins == i, pidx_bins == j))[0]
            t += len(dic[i][j])

    return dic, score, p, vbins, pbins
