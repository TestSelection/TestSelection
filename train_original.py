'''
The file is used to train the original modles.
'''
from numpy.random import seed
seed(54345)
from tensorflow import set_random_seed
set_random_seed(65466756)
import exp_models.cifar.Network_in_Network as netinnet
import exp_models.cifar.VGG16_Clipped as vgg
import exp_models.mnist.deepxplore as deepxplore
import exp_models.mnist.letnet as letnet
import exp_models.mnist.mlp as mlp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="mnist", metavar='STRING',
                        help='dataset. (default: mnist)')
    parser.add_argument('-m', '--model', type=str, default="mlp", metavar='STRING',
                        help='model. (default: mlp)')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='dataset. (default: mnist)')
    args = parser.parse_args()
    modelname =  args.model
    dataset =  args.dataset


    if modelname == 'netinnet':
        netinnet.train(dataset, epochs= 300)

    if modelname == "vgg":
        vgg.train_VGG16(dataset, num_epoch= 300)

    if modelname == "deepxplore":
        deepxplore.train(dataset, epochs= 50)

    if modelname == "lenet":
        letnet.train(dataset, epochs= 50)

    if modelname == "mlp":
        mlp.train(dataset, epochs= 50)
