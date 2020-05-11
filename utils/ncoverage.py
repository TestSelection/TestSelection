import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Convolution2D

'''
The code is from DeepTest: Automated testing of deep-neural-network-driven autonomous cars
https://github.com/ARiSE-Lab/deepTest.
We modeify the code to implement the neuron coverage in DeepGauge.
'''
class NCoverage():

    def __init__(self, model, threshold=0.1, only_layers=[]):
        '''
        Initialize the model to be tested
        :param threshold: threshold to determine if the neuron is activated
        :param model_name: ImageNet Model name, can have ('vgg16','vgg19','imagenet')
        :param neuron_layer: Only these layers are considered for neuron coverage
        '''
        self.threshold = float(threshold)
        self.model = model
        print('models loaded')

        # the layers that are considered in neuron coverage computation
        #self.layer_to_compute = []
        #for layer in self.model.layers:
        #    if len(layer.get_weights())>0:
        #        self.layer_to_compute.append(layer.name)
        self.layer_to_compute = []
        for layer in self.model.layers:
            if len(layer.get_weights()) > 0:
                # print(type(layer))
                if isinstance(layer, Dense) or isinstance(layer, Conv2D) or  isinstance(layer, Convolution2D):
                    self.layer_to_compute.append(layer.name)



        if len(only_layers)!=0:
            self.layer_to_compute = only_layers
        print(self.layer_to_compute)
        # init coverage table
        self.cov_dict = {}
        self.kmnc ={}
        self.outputstd = {}
        for layer_name in self.layer_to_compute:
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                self.cov_dict[(layer_name, index)] = False
                self.kmnc[(layer_name, index)] = (None, None)
                self.outputstd[(layer_name, index)] = None



    def NC(self, input, **kwargs):
        '''
        Compute set nc score
        :param input:
        :param kwargs:
        :return:
        '''
        # init coverage table
        self.reset_cov_dict()
        self.update_coverage(input)
        _,_,nc = self.curr_neuron_cov()
        return nc

    def NBC(self, input, both=False,**kwargs):
        '''
        compute set bnc score
        :param input:
        :return:
        '''
        nbc_count = 0
        snac_count = 0
        total = 0
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input) # num_samples, features, neurons
            for neron_idx in range(layer_outputs.shape[-1]):
                    (low, high) = self.kmnc[(layer_name, neron_idx)]
                    total += 1
                    batch_outs = layer_outputs[..., neron_idx] # num_samples, features
                    batch_outs = np.reshape(batch_outs, (layer_outputs.shape[0], -1))
                    out = np.mean(batch_outs, axis=1)  # num_samples
                    nbc_count += 1 if np.any(out < low) else 0
                    snac_count += 1 if np.any(out > high) else 0
            del layer_model
            del layer_outputs

        if both:
            return (snac_count+nbc_count)/(2.0*total),snac_count/total
        else:
            return (snac_count+nbc_count) / (2.0 * total)

    def SNAC(self, input,**kwargs):
        '''
        compute set sanc score
        :param input:
        :return:
        '''
        snac_count = 0
        total = 0
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input, batch_size=2)  #num_samples, features, neurons
            for neron_idx in range(layer_outputs.shape[-1]):
                    (low, high) = self.kmnc[(layer_name, neron_idx)]
                    total += 1
                    batch_outs = layer_outputs[..., neron_idx] #num_samples, features,
                    batch_outs = np.reshape(batch_outs, (layer_outputs.shape[0], -1))
                    out = np.mean(batch_outs, axis=1) #num_samples,
                    snac_count+=1 if np.any(out > high) else 0
            del layer_model
            del layer_outputs

        return snac_count /total



    def KMNC(self, input_x, K=1000,onlyKMNC=True, **kwargs):
        '''
        compute set kmnc score
        :param input: Single Input. KMNC degrades to judge whether otuput is in [low, high]
        :param K:
        :return:
        '''
        kmnc_count = 0
        total = 0
        upperCorner = 0
        lowerCorner = 0
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_x, batch_size=1)
            for neron_idx in range(layer_outputs.shape[-1]):
                    total += 1
                    (low, high) = self.kmnc[(layer_name, neron_idx)]  #get low and high boundary
                    batch_outs = layer_outputs[..., neron_idx]        #get outputs for each neuron
                    batch_outs = np.reshape(batch_outs, (layer_outputs.shape[0], -1)) #(num_images, features_output)
                    out = np.mean(batch_outs, axis=1)  #compute neuron output for each image #(num_images,
                    upperCorner += 1 if np.any(out>high) else 0
                    lowerCorner += 1 if np.any(out<low) else 0
                    buckets = np.floor((out - low)[np.logical_and(out<high, out>low)]/(high-low)*K,)
                    buckets = buckets.astype(int)
                    c = np.bincount(buckets)
                    kmnc_count += np.count_nonzero(c)
            del layer_model
            del layer_outputs
        if onlyKMNC:
            return kmnc_count/(total*K)*1.0
        else:
            return kmnc_count / (total * K) * 1.0, (upperCorner + lowerCorner)/(2.0*total), upperCorner/total*1.0

    def batch_kmnc(self, input_data, K=1000):
        '''
                score each image
                :param input: Single Input. KMNC degrades to judge whether otuput is in [low, high]
                :param K:
                :return:
                '''
        kmnc_count = []
        total = 0
        upperCorner = []
        lowerCorner = []
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data) # (num_samples, features, neuron_idx)
            total += layer_outputs.shape[-1]
            reshape_layerouputs = layer_outputs.reshape(layer_outputs.shape[0], -1, layer_outputs.shape[-1]) # (num_samples, feature, neuron_idx)
            neruonsoutput = np.squeeze(np.mean(reshape_layerouputs, axis=1)) # (num_samples, neurons)
            max_vector, min_vector = self.kmnc_batch[layer_name]['max'],self.kmnc_batch[layer_name]['min']
            upperCornerNeuron = np.sum(neruonsoutput > max_vector, axis=1) #(num_samples,
            lowerCornerNeuron = np.sum(neruonsoutput < min_vector, axis=1) # (num_samples,
            upperCorner.append(upperCornerNeuron)
            lowerCorner.append(lowerCornerNeuron)
            validindex = np.logical_and(neruonsoutput < max_vector, neruonsoutput > min_vector) # fail in the range
            bucket = np.sum(validindex, axis=1) # (num_samples )
            kmnc_count.append(bucket)
            del layer_model
            del layer_outputs,bucket,validindex
            del reshape_layerouputs, neruonsoutput,max_vector, min_vector


        kmnc_count = np.asarray(kmnc_count) # num_layer,num_samples
        upperCorner = np.asarray(upperCorner)# num_layer,num_samples
        lowerCorner = np.asarray(lowerCorner)# num_layer,num_samples

        return (np.sum(kmnc_count, axis=0)/total,np.sum(upperCorner+lowerCorner, axis=0)/(2*total),
                np.sum(upperCorner, axis=0)/total)


    def initKMNCtable(self,input_data, file, read=False,save=True):
        '''
        Init k-multisection Neuron Coverage using the training data
        :param input_data:
        :return:
        '''
        import pickle
        import os
        if read and os.path.isfile(file):
            with open(file, 'rb') as fp:
                data = pickle.load(fp)
                self.kmnc, self.outputstd, self.covered_neruons,self.kmnc_batch,  self.std_batch = data['kmnc'], data['std'], data['coverednuron'],data['kmnc_batch'],data['std_batch']
                return
        self.covered_neruons = {} # layer_name:vector
        self.kmnc_batch = {} # layer_name:{min: neruons, max: neruons}
        self.std_batch = {} # layer_name:std
        for layer_name in self.layer_to_compute:
            self.kmnc_batch[layer_name] = {}
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data)
            reshape_layerouputs = layer_outputs.reshape(layer_outputs.shape[0], -1,
                                                        layer_outputs.shape[-1]) # num_sample, features, neurons
            reshape_layerouputs = np.mean(reshape_layerouputs, axis=1)# num_sample, neurons
            neuron_max = np.squeeze(np.max(reshape_layerouputs, axis=0)) #neurons
            self.kmnc_batch[layer_name]["max"]=neuron_max
            neuron_min = np.squeeze(np.min(reshape_layerouputs, axis=0))#neurons
            self.kmnc_batch[layer_name]["min"] = neuron_min
            neuron_std = np.squeeze(np.var(reshape_layerouputs, axis=0))#neurons
            self.std_batch[layer_name]=neuron_std

            #neruons = np.zeros(layer_outputs.shape[-1])
            scaled = self.batch_scale(layer_outputs)
            ncc = scaled.reshape(scaled.shape[0], -1, scaled.shape[-1])  # num_sample, features, neurons
            nccout = np.mean(ncc, axis=1)  # num_sample, neurons

            self.covered_neruons[layer_name] = np.any(nccout > self.threshold, axis=0)+0 #neurons

            for neron_idx in range(layer_outputs.shape[-1]):
                    batch_outs = layer_outputs[..., neron_idx]
                    batch_outs = np.reshape(batch_outs, (batch_outs.shape[0], -1))
                    out = np.mean(batch_outs, axis=1)    #compute the neuron's mean for each image
                    lown, highn = np.min(out), np.max(out)
                    nstd = np.var(out)  #compute the neuron's variance
                    self.outputstd[(layer_name, neron_idx)] = nstd

                    if (layer_name, neron_idx) in self.kmnc:
                        (low, high) = self.kmnc[(layer_name, neron_idx)]
                        if low== None or  lown <low:
                            low = lown
                        if high==None or highn >high:
                            high = highn
                        self.kmnc[(layer_name, neron_idx)] = (low, high)
                    else:
                        self.kmnc[(layer_name, neron_idx)] = (0, 0)

            del layer_model
            del layer_outputs
            del reshape_layerouputs
            del neuron_max, neuron_min,neuron_std,scaled,ncc,nccout
        #self.init_nc_table(input_data)
        if(save):
            with open(file, 'wb') as fp:
                pickle.dump({"kmnc":self.kmnc, "std":self.outputstd,
                             "coverednuron":self.covered_neruons, "kmnc_batch":self.kmnc_batch, "std_batch":self.std_batch}, fp, protocol=pickle.HIGHEST_PROTOCOL)


    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def batch_scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        shape = layer_outputs.shape
        reshapelayer_outputs = layer_outputs.reshape(layer_outputs.shape[0], -1) # num_smapple, features*neurons

        divider = np.max(reshapelayer_outputs, axis=1) - np.min(reshapelayer_outputs, axis=1)
        #if divider == 0:
        #    return np.zeros(shape=layer_outputs.shape)
        divider[divider==0] = 1
        divider = divider.reshape(-1,1)
        X_std = (reshapelayer_outputs - np.min(reshapelayer_outputs, axis=1).reshape(-1,1)) / divider
        X_std = X_std.reshape(shape)
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def set_covdict(self, covdict):
        self.cov_dict = dict(covdict)

    def batch_nc(self, input_data):
        '''
        Compute NC for each image
        :param input_data:
        :return:
        '''
        res = []
        total = 0
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data) #num samples, features, neurons
            total += layer_outputs.shape[-1]
            scaled = self.batch_scale(layer_outputs) # 0-1
            scaled = scaled.reshape(scaled.shape[0],-1, scaled.shape[-1]) #num samples, features, neurons
            neuron_output = np.mean(scaled, axis=1) #num samples, neurons
            covered = np.sum(neuron_output>self.threshold, axis=1) #num samples,
            res.append(covered)
            del layer_outputs
            del layer_model
            del scaled
            del neuron_output
            del covered

        res = np.asarray(res)
        n = np.sum(res, axis=0)
        nc = n/total
        del res
        del n
        return nc

    def update_coverage(self, input_data):
        '''
        Given the input, update the neuron covered in the model by this input.
            This includes mark the neurons covered by this input as "covered"
        :param input_data: the input image
        :return: the neurons that can be covered by the input
        '''
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data)
            for layer_output in layer_outputs:
                scaled = self.scale(layer_output)
                #print(scaled.shape)
                for neuron_idx in range(scaled.shape[-1]):
                    if np.mean(scaled[..., neuron_idx]) > self.threshold:
                        self.cov_dict[(layer_name, neuron_idx)] = True
            del layer_outputs
            del layer_model
        return self.cov_dict

    # def init_nc_table(self, input_data):
    #     '''
    #     Given Input, label the covered neurons.
    #     {layer_name:vector}
    #     :param init_data:
    #     :return:
    #     '''
    #     self.covered_neruons = {}
    #     for layer_name in self.layer_to_compute:
    #         layer_model = Model(inputs=self.model.inputs,
    #                            outputs=self.model.get_layer(layer_name).output)
    #         layer_outputs = layer_model.predict(input_data)
    #         neruons = np.zeros(layer_outputs.shape[-1])
    #         for layer_output in layer_outputs:
    #             scaled = self.scale(layer_output)
    #             for neuron_idx in range(scaled.shape[-1]):
    #                 if np.mean(scaled[..., neuron_idx]) > self.threshold:
    #                     neruons[neuron_idx] = 1
    #         self.covered_neruons[layer_name] = neruons

    def diffScore(self, input_data):
        '''

        :param input_data:
        :return:
        '''
        total = 0
        diff = 0
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                               outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data)

            scaled = self.batch_scale(layer_outputs)
            scaled = scaled.reshape(scaled.shape[0], -1, scaled.shape[-1])
            nc_covered = np.mean(scaled, axis=1) > self.threshold  # num_samples, neurons
            coveredneurons = np.any(nc_covered, axis=0) + 0 # neurons
            total += np.sum(coveredneurons)
            diff += np.count_nonzero((coveredneurons - self.covered_neruons[layer_name])>0)  # covered new neurons
            del layer_model
            del layer_outputs
            del scaled
            del nc_covered
            del coveredneurons
        if total==0:
            return 0
        return diff/total

    def batch_diffScore(self, input_data):
        '''
        Given Input_data and init covered neruons, compute the different covered neurons
        :param input_data:
        :return:
        '''

        input_nc = []
        total = np.zeros(input_data.shape[0])
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                               outputs=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data) # num samples, features, num_neurons

            scaled = self.batch_scale(layer_outputs)
            scaled = scaled.reshape(scaled.shape[0], -1, scaled.shape[-1]) # num samples, features, num_neurons
            nc_covered = np.mean(scaled, axis=1)>self.threshold + 0 # num_samples, neurons
            total += np.sum(nc_covered, axis=1)
            diff = np.count_nonzero(((nc_covered - self.covered_neruons[layer_name])>0+0) , axis=1) # num_samples
            input_nc.append(diff)
            del diff
            del scaled
            del nc_covered
            del layer_outputs
            del layer_model

        input_nc = np.asarray(input_nc) # num_layers, num_samples
        total[total==0] = 1
        score = np.sum(input_nc, axis=0)/total
        del input_nc
        del total
        return score


    def curr_neuron_cov(self):
        '''
        Get current coverage information of MUT
        :return: number of covered neurons,
            number of total neurons,
            number of neuron coverage rate
        '''
        covered_neurons = len([v for v in self.cov_dict.values() if v])
        total_neurons = len(self.cov_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)




    def reset_cov_dict(self):
        '''
        Reset the coverage table
        :return:
        '''
        for layer_name in self.layer_to_compute:
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                self.cov_dict[(layer_name, index)] = False
