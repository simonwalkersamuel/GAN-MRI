# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:52:42 2018
@author: simon
"""
import numpy as np
import random
import io as inout
import nibabel as nib
import os
join = os.path.join
import datetime
import sys
import nibabel as nib
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
import glob
from functools import partial
from skimage import io

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, Flatten, Dense, Reshape, Lambda, Add, UpSampling3D
from tensorflow.keras.layers import Layer, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
import tensorflow.keras as keras
from tensorflow.keras import backend
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from collections import OrderedDict

import CycleGAN3d

# Set the logger
LPATH = '/mnt/data2/retinasim/retina_cyclegan_output/log'
if False:
    from logger import Logger
    if not os.path.exists(LPATH):
        os.mkdir(LPATH)
    D_A_log_dir = join(LPATH,'D_A_logs')
    D_B_log_dir = join(LPATH,'D_B_logs')
    if not os.path.exists(D_A_log_dir):
        os.mkdir(D_A_log_dir)
    D_A_logger = Logger(D_A_log_dir)
    if not os.path.exists(D_B_log_dir):
        os.mkdir(D_B_log_dir)
    D_B_logger = Logger(D_B_log_dir)

    G_A_log_dir = join(LPATH,'G_A_logs')
    G_B_log_dir = join(LPATH,'G_B_logs')
    if not os.path.exists(G_A_log_dir):
        os.mkdir(G_A_log_dir)
    G_A_logger = Logger(G_A_log_dir)
    if not os.path.exists(G_B_log_dir):
        os.mkdir(G_B_log_dir)
    G_B_logger = Logger(G_B_log_dir)

    cycle_A_log_dir = join(LPATH,'cycle_A_logs')
    cycle_B_log_dir = join(LPATH,'cycle_B_logs')
    if not os.path.exists(cycle_A_log_dir):
        os.mkdir(cycle_A_log_dir)
    cycle_A_logger = Logger(cycle_A_log_dir)
    if not os.path.exists(cycle_B_log_dir):
        os.mkdir(cycle_B_log_dir)
    cycle_B_logger = Logger(cycle_B_log_dir)

    img_log_dir = join(LPATH,'img_logs')
    if not os.path.exists(img_log_dir):
        os.mkdir(img_log_dir)
    img_logger = Logger(img_log_dir)


# weighted sum output
class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output
        
    def get_config(self):

        config = super().get_config().copy()
        #config.update({
        #    'alpha': self.alpha,
        #})
        return config

# mini-batch standard deviation layer
class MinibatchStdev(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
 
    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], shape[3], 1))
        # concatenate with the output
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined
 
    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)
        
    def get_config(self):

        config = super().get_config().copy()
        #config.update({
        #    'alpha': self.alpha,
        #})
        return config

# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
 
    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs**2.0
        # calculate the mean pixel values
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized
 
    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape        
        
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred) 
    
def parse_directory(path):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".nii"):
                res.append(join(root,file))
    return res
    
def load_real_image(filename):
    # load dataset
    img = nib.load(filename)
    im = np.array(img.dataobj)
    return im
    
def load_sim_image(filename):
    # load dataset
    img = nib.load(filename)
    im = np.array(img.dataobj)
    im[im>0] = 1
    return im
    
def load_real_samples(path,n=None):
    files = parse_directory(path)
    im = []
    for i,file in enumerate(files):
        im.append(load_real_image(file))
        if n is not None and i>=(n-1):
            break
    im = np.asarray(im)
    # convert from ints to floats
    im = im.astype('float32')
    # scale from [0,255] to [-1,1]
    im = (im - 127.5) / 127.5
    return im
    
def load_sim_samples(path,n=None):
    files = parse_directory(path)
    im = []
    for i,file in enumerate(files):
        im.append(load_sim_image(file))
        if n is not None and i>=(n-1):
            break
    im = np.asarray(im)
    # convert from ints to floats
    im = im.astype('float32')
    # scale from [0,255] to [-1,1]
    im = (im - 127.5) / 127.5
    return im    
        
class TubeProgCycleGAN(object):

    def __init__(self, opath=''):
        self.progressive = True
        self.generators = None
        self.discriminators = None
        self.composite = None
        self.opath = opath

        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 2000  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 50

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = True

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        if date_time_string == '' and not resume:
            self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition
        else:
            self.date_time = date_time_string + date_time_string_addition

    def weight_initialisation(self):
        return RandomNormal(stddev=0.02)
        
    def weight_constraint(self):
        return max_norm(1.0)
        
# UNET GENERATOR-----    

    def generator_encode_block(self,input_layer,n,input_shape=None,no_pool=False,index=0,add_skips=True):

        init = self.weight_initialisation()
        const = self.weight_constraint()
        
        if input_shape is not None:
            conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,input_shape=input_shape,name='conv_encode{}_{}_{}'.format(index,n,1))(input_layer)
        else:
            conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_encode{}_{}_{}'.format(index,n,1))  (input_layer)

        dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)            
        #conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_encode{}_{}_{}'.format(index,n,2))(dr)
        #dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)
        
        if no_pool:
            return dr,conv
        pool = MaxPooling3D(pool_size=(2, 2, 2))(dr)
        return pool,conv
        
    def generator_decode_block(self,input_layer,n,conv_layer=None,index=0,add_skips=True,level=0):
        
        init = self.weight_initialisation()
        const = self.weight_constraint()
        
        convTr_layer = Conv3DTranspose(n, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_layer)
        if add_skips:
            up = concatenate([convTr_layer, conv_layer], axis=4)
        else:
            up = convTr_layer

        conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_decode{}_{}_{}'.format(index,n,1))(up)
        dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)
        #conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_decode{}_{}_{}'.format(index,n,2))(dr)
        #dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)
        return dr      

    # define generator models
    def create_progressive_generator(self,size=512,nchannel=1,max_blocks=15,add_skips=True,nblock=None,name='cGAN_gen'):
    
        init = self.weight_initialisation()
        const = self.weight_constraint()
        
        model_list = []
        
        if nblock is None:
            nblockLrg = int(np.log(size) / np.log(2))
            nblock = np.min([max_blocks,nblockLrg])
            
        # Number of filters on each layer
        ns = np.asarray([np.power(2,x+1) for x in range(2,nblock+2)])
        ns = np.clip(ns,0,512)
        # FIX NS
        ns = np.zeros(nblock+1,dtype='int')+128
        # Image dims
        ld = np.asarray([np.power(2,x+1) for x in range(nblock+1)])
        # Reverse order
        ld = ld[::-1]
        
        conv_layers = []
        encode_out = []
        decode_out = []

        for i in range(nblock,0,-1):
            print(i,nblock-1,ld[i])
        
            # Input layer
            input_shape = (ld[i],ld[i],ld[i],nchannel)
            input_layer = Input(shape=input_shape)
            # Linear scale-up
            dl = Dense(128, kernel_initializer=init, kernel_constraint=const)(input_layer)
            #dl = Reshape(input_shape[0:3]+(128,))(dl)
            
            # Encoding
            no_pool = False # i==(nblock-1) # Don't use pooling on last encoding layer
            #if i==0:
            #    enc_out,conv = self.generator_encode_block(dl,ns[i],input_shape=input_shape,no_pool=no_pool,index=i+1)
            #else:
            if True:
                enc_out,conv = self.generator_encode_block(dl,ns[i],no_pool=no_pool,index=i+1)
            conv_layers.append(conv)
            
            # Append previous model
            out = enc_out
            if i<nblock:
                old_model = model_list[-1][1]
                #if i==0:
                #    import pdb
                #    pdb.set_trace()
                for layer1 in old_model.layers[2:]: # Skip input layers
                    try:
                        out = layer1(out)
                    except Exception as e:
                        print(e)
                        import pdb
                        pdb.set_trace()
            
            # Straight-through decoder
            add_skips = False
            if i!=nblock:
                db = self.generator_decode_block(out,ns[i],conv_layer=conv_layers[-1],index=i+1,add_skips=add_skips,level=i)
            else:
                db = self.generator_decode_block(out,ns[i],index=i+1,add_skips=False,level=i)
                
            # Final layer
            dec_out = Conv3D(1, 1, activation='sigmoid')(db)
            decode_out.append(dec_out)
                 
            # Straight-through model   
            model2 = Model(input_layer,dec_out,name=name)
            model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)) 
            
            if i<nblock:
                # Add grade-in weighted sum blocks
                model1 = Model()
                input_layer = Input(shape=input_shape)
                dl = model2.layers[1](input_layer)
                
                mp = MaxPooling3D(pool_size=(2, 2, 2))(dl)
                
                ninputlayers = 2
                nenclayers = 3
                if add_skips:
                    ndeclayers = 4
                else:
                    ndeclayers = 3                
                noutlayers = 1
                
                # Encoding
                d = dl
                layer_count = ninputlayers
                for layer1 in model2.layers[ninputlayers:ninputlayers+nenclayers]:
                    d = layer1(d)
                    layer_count += 1

                d = WeightedSum()([mp, d])
                
                for l in range(ninputlayers+nenclayers, len(model2.layers)-(ndeclayers+noutlayers)-1):
                    layer1 = model2.layers[l]
                    d = layer1(d)
                    
                mpd = UpSampling3D(size=(2, 2, 2))(d)
                
                # Decoding
                l0,l1 = len(model2.layers)-ndeclayers-noutlayers-1, len(model2.layers)-noutlayers-1
                for layer1 in model2.layers[l0:l1]:
                    d = layer1(d)
                    
                out = WeightedSum()([mpd, d])
                
                # Final layer
                d = out
                for layer1 in model2.layers[-noutlayers:]:
                    d = layer1(d)
                out = d
                
                model1 = Model(input_layer,out,name=name+'_fadeIn')
                # Not compiled as generator not trained directly
                #model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)) 
            else:
                model1 = model2
                
            model_list.append([model1,model2])
        
        self.generators = model_list
        return model_list
                    
# DISCRIMINATOR-----
            
    def discriminator_input_model(self,input_shape=(4,4,4,1)):
        init = self.weight_initialisation()
        const = self.weight_constraint()
        
        # base model input
        in_image = Input(shape=input_shape)
        # conv 1x1
        d = Conv3D(128, (1,1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)

        return d,in_image

    # define the discriminator models for each image resolution
    def create_progressive_discriminator(self,nblock=None,name=None):
    
        init = self.weight_initialisation()
        const = self.weight_constraint()
    
        model_list = []
        
        # Smallest model first
        # Input layers
        d,in_image = self.discriminator_input_model(input_shape=(2,2,2,1))
        
        # Output block
        # conv 3x3
        d = MinibatchStdev()(d)
        d = Conv3D(128, (3,3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # conv 4x4
        #d = Conv3D(128, (4,4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        #d = LeakyReLU(alpha=0.2)(d)
        
        # dense output layer
        d = Flatten()(d)
        out_class = Dense(1)(d)
        # define model
        model = Model(in_image, out_class, name=name)
        # compile model
        model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)) 
        
        # store model
        model_list.append([model,model])
        
        # create submodels
        for i in range(1, nblock):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            models = self.add_discriminator_block(old_model)
            # store model
            model_list.append(models)
            
        self.discriminators = model_list
        return model_list 
        
    def discriminator_block(self,d):
    
        init = self.weight_initialisation()
        const = self.weight_constraint()
            
        # define new block
        d = Conv3D(128, (3,3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        #d = Conv3D(128, (3,3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        #d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling3D()(d)
        return d        
        
    # add a discriminator block to progressive GAN
    def add_discriminator_block(self, old_model, n_input_layers=3):
    
        init = self.weight_initialisation()
        const = self.weight_constraint()
        
        # get shape of existing model
        in_shape = list(old_model.input.shape)
        
        # define new input shape as double the size
        input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
        in_d, in_image = self.discriminator_input_model(input_shape=input_shape)
        d = self.discriminator_block(in_d)
    
        # Tack on layers from previous model
        block_new = d
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        # define straight-through model
        model1 = Model(in_image, d)
        # compile model
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # downsample the new larger image
        downsample = AveragePooling3D()(in_image)
        # connect old input processing to downsampled new input
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        # fade in output of old model input layer with new input
        d = WeightedSum()([block_old, block_new])
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        # define straight-through model
        model2 = Model(in_image, d)
        # compile model
        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        return [model1, model2]
        
# COMPOSITE----        

    # define composite models for training generators via discriminators
    def create_progressive_composite(self):
    
        if self.generators is None or self.discriminators is None:
            return None
            
        model_list = []

        # create composite models
        for i in range(len(self.discriminators)):
            print('Composite it: {}'.format(i))
            g_models, d_models = self.generators[i], self.discriminators[i]
            # straight-through model
            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            try:
                model1.add(d_models[0])
            except:
                import pdb
                pdb.set_trace()
            model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            # fade-in model
            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            # store
            model_list.append([model1, model2])
        self.composite = model_list
        return model_list   
        
#TRAIN-----     

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, input_images, n_samples):
        # generate points in latent space
        #x_input = generate_latent_points(latent_dim, n_samples)
        # predict outputs
        gen_ims = []
        n_samples = input_images.shape[0]
        X = generator.predict(input_images)
        # create class labels
        y = -np.ones((n_samples, 1))
        return X, y      
        
    def generate_real_samples(self, dataset, n_samples):
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        # select images
        X = dataset[ix]
        # generate class labels
        y = np.ones((n_samples, 1))
        return X, y    
        
    # update the alpha value on each instance of WeightedSum
    def update_fadein(self, models, step, n_steps):
        # calculate current alpha (linear from 0 to 1)
        alpha = step / float(n_steps - 1)
        # update the alpha for each model
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    backend.set_value(layer.alpha, alpha)

    # scale images to preferred size
    def scale_dataset(self, images, new_shape):
        print('Sclaing images: {}'.format(new_shape))
        images_list = []
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)

    def summarize_performance(self,status, g_model, scaled_sim_data, n_samples=25):
        # devise name
        gen_shape = g_model.output_shape
        name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
        # generate images
        X, _ = self.generate_unet_fake_samples(g_model, scaled_sim_data[0:1], n_samples)
        # normalize pixel values to the range [0,1]
        X = (X - X.min()) / (X.max() - X.min())
        # plot real images
        #square = int(sqrt(n_samples))
        #for i in range(n_samples):
        #    pyplot.subplot(square, square, 1 + i)
        #    pyplot.axis('off')
        #    pyplot.imshow(X[i])
        # save plot to file
        filename1 = join(self.opath,'sample_{}.nii'.format(name))
        img = nib.Nifti1Image(np.clip(X[0],0,1),np.eye(4))
        nib.save(img,filename1)
        #pyplot.savefig(filename1)
        #pyplot.close()
        # save the generator model
        filename2 = join(self.opath,'model_{}.h5'.format(name))
        g_model.save(filename2)
        print('>Saved: {}'.format(filename2))
                        
    # train a generator and discriminator
    def train_epochs(self,g_model, d_model, gan_model, real_data, sim_data, n_epochs, n_batch, fadein=False):
        # calculate the number of batches per training epoch
        bat_per_epo = int(real_data.shape[0] / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_steps):
            # update alpha for all WeightedSum layers when fading in new blocks
            if fadein:
                self.update_fadein([g_model, d_model, gan_model], i, n_steps)
            # prepare real and fake samples
            X_real, y_real = self.generate_real_samples(real_data, half_batch)
            X_fake, y_fake = self.generate_unet_fake_samples(g_model, sim_data, half_batch)
            # update discriminator model
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # update the generator via the discriminator's error
            z_input = sim_data[0:n_batch] # generate_latent_points(latent_dim, n_batch)
            y_real2 = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(z_input, y_real2)
            # summarize loss on this batch
            print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))    
            
    def train(self, g_models, d_models, gan_models, real_images, sim_images, e_norm, e_fadein, n_batch):       
        # fit the baseline model
        g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
        # scale dataset to appropriate size
        gen_shape = g_normal[0].output_shape
        
        scaled_real_data = self.scale_dataset(real_images, gen_shape[1:])
        scaled_sim_data = self.scale_dataset(sim_images, gen_shape[1:])
        
        print('Scaled Data', scaled_real_data.shape)
        
        # train normal or straight-through models
        #self.train_epochs(g_normal, d_normal, gan_normal, scaled_real_data, scaled_sim_data, e_norm[0], n_batch[0])
        self.G_A2B = g_normal[0]
        self.G_B2A = g_normal[1]
        self.D_A2B = d_normal[0]
        self.D_B2A = d_normal[1]
        self.train_cgan(gan_normal, scaled_real_data, scaled_sim_data, e_norm[0], n_batch[0])
        self.summarize_performance('tuned', g_normal, scaled_sim_data)
        # process each level of growth
        for i in range(1, len(g_models)):
            # retrieve models for this level of growth
            [g_normal, g_fadein] = g_models[i]
            [d_normal, d_fadein] = d_models[i]
            [gan_normal, gan_fadein] = gan_models[i]
            # scale dataset to appropriate size
            gen_shape = g_normal[0].output_shape
            scaled_real_data = self.scale_dataset(real_images, gen_shape[1:])
            scaled_sim_data = self.scale_dataset(sim_images, gen_shape[1:])
            print('Scaled Data', scaled_real_data.shape)
            # train fade-in models for next level of growth
            self.train_unet_epochs(g_fadein, d_fadein, gan_fadein, scaled_real_data, scaled_sim_data, e_fadein[i], n_batch[i], True)
            self.summarize_unet_performance('faded', g_fadein, scaled_sim_data)
            # train normal or straight-through models
            self.G_A2B = g_normal[0]
            self.G_B2A = g_normal[1]
            #self.train_epochs(g_normal, d_normal, gan_normal, scaled_real_data, scaled_sim_data, e_norm[i], n_batch[i])
            self.train_cgan(g_normal, d_normal, gan_normal, scaled_real_data, scaled_sim_data, e_norm[i], n_batch[i])
            self.summarize_unet_performance('tuned', g_normal, scaled_sim_data)   
            
#===============================================================================
# CycleGAN Training
    def train_cgan(self, epochs, real_images_A, real_images_B, synthetic_images_A, synthetic_images_B, batch_size=1, save_interval=1):
    
        #tbCallBack = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=False, write_images=False)
        
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
                # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                if self.use_multiscale_discriminator:
                    DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                    DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                    print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                    print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
                else:
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones)
                target_data.append(ones)

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                G_A2B_identity_loss = self.G_A2B.train_on_batch(
                    x=real_images_B, y=real_images_B)
                G_B2A_identity_loss = self.G_B2A.train_on_batch(
                    x=real_images_A, y=real_images_A)
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            if loop_index % 20 == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # labels
        if self.use_multiscale_discriminator:
            label_shape1 = (batch_size,) + self.D_A.output_shape[0][1:]
            label_shape2 = (batch_size,) + self.D_A.output_shape[1][1:]
            #label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            #zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (batch_size,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_images_A = images[0]
                    real_images_B = images[1]

                    if len(real_images_A.shape) == 4:
                        real_images_A = real_images_A[:, :, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, :, np.newaxis]

                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator.__len__())
                    
                    # ============ TensorBoard logging ============#
                    #print('Tensorboard:',DA_loss,DB_loss,gA_d_loss_synthetic,gB_d_loss_synthetic,reconstruction_loss_A,reconstruction_loss_B)
                    D_A_logger.scalar_summary('losses', DA_losses[-1], loop_index)
                    D_B_logger.scalar_summary('losses', DB_losses[-1], loop_index)
                    G_A_logger.scalar_summary('losses', gA_d_losses_synthetic[-1], loop_index)
                    G_B_logger.scalar_summary('losses', gB_d_losses_synthetic[-1], loop_index)
                    #cycle_A_logger.scalar_summary('losses', reconstruction_losses[-1], loop_index)
                    #cycle_B_logger.scalar_summary('losses', reconstruction_losses_B[-1], loop_index)

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__() or loop_index>=self.images_per_epoch:
                        break

                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.A_train
                B_train = self.B_train
                random_order_A = np.random.randint(len(A_train), size=len(A_train))
                random_order_B = np.random.randint(len(B_train), size=len(B_train))
                epoch_iterations = max(len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                # If we want supervised learning the same images form
                # the two domains are needed during each training iteration
                if self.use_supervised_learning:
                    random_order_B = random_order_A
                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            indexes_A = np.random.randint(len(A_train), size=batch_size)

                            # if all images are used for the other domain
                            if loop_index + batch_size >= epoch_iterations:  
                                indexes_B = random_order_B[epoch_iterations-batch_size: 
                                                           epoch_iterations]
                            else: # if not used, continue iterating...
                                indexes_B = random_order_B[loop_index:
                                                           loop_index + batch_size]

                        else: # if len(B_train) <= len(A_train)
                            indexes_B = np.random.randint(len(B_train), size=batch_size)
                            
                             # if all images are used for the other domain
                            if loop_index + batch_size >= epoch_iterations:  
                                indexes_A = random_order_A[epoch_iterations-batch_size: 
                                                           epoch_iterations]
                            else: # if not used, continue iterating...
                                indexes_A = random_order_A[loop_index:
                                                           loop_index + batch_size]

                    else:
                        indexes_A = random_order_A[loop_index:
                                                   loop_index + batch_size]
                        indexes_B = random_order_B[loop_index:
                                                   loop_index + batch_size]

                    sys.stdout.flush()
                    real_images_A = A_train[indexes_A]
                    real_images_B = B_train[indexes_B]

                    # Run all training steps
                    D_A_loss,D_B_loss,G_A_loss = run_training_iteration(loop_index, epoch_iterations)
                    
                    # ============ TensorBoard logging ============#


            #================== within epoch loop end ==========================

            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)

            if epoch % 20 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_A, epoch)
                self.saveModel(self.D_B, epoch)
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()   
            

#===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 4:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 5:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, :, 0]

        #toimage(image, cmin=-1, cmax=1).save(path_name)
        img = nib.Nifti1Image(np.clip(image,-1,1),np.eye(4))
        nib.save(img,path_name)

    def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
        directory = os.path.join(OPATH,'images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))
            os.makedirs(os.path.join(directory, 'Atest'))
            os.makedirs(os.path.join(directory, 'Btest'))

        testString = ''

        real_image_Ab = None
        real_image_Ba = None
        for i in range(num_saved_images + 1):
            if i == num_saved_images:
                real_image_A = self.A_test[0]
                real_image_B = self.B_test[0]
                real_image_A = np.expand_dims(real_image_A, axis=0)
                real_image_B = np.expand_dims(real_image_B, axis=0)
                testString = 'test'
                if self.channels == 1:  # Use the paired data for MR images
                    real_image_Ab = self.B_test[0]
                    real_image_Ba = self.A_test[0]
                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)
            else:
                #real_image_A = self.A_train[rand_A_idx[i]]
                #real_image_B = self.B_train[rand_B_idx[i]]
                if len(real_image_A.shape) < 4:
                    real_image_A = np.expand_dims(real_image_A, axis=0)
                    real_image_B = np.expand_dims(real_image_B, axis=0)
                if self.channels == 1:  # Use the paired data for MR images
                    real_image_Ab = real_image_B  # self.B_train[rand_A_idx[i]]
                    real_image_Ba = real_image_A  # self.A_train[rand_B_idx[i]]
                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)

            try:
                synthetic_image_B = self.G_A2B.predict(real_image_A)
                synthetic_image_A = self.G_B2A.predict(real_image_B)
                reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
                reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)
            except Exception as e:
                print(e)
                return

            self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
                                 os.path.join(OPATH,'images/{}/{}/epoch{}_sample{}.nii'.format(
                                     self.date_time, 'A' + testString, epoch, i)))
            self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
                                 os.path.join(OPATH,'images/{}/{}/epoch{}_sample{}.nii'.format(
                                     self.date_time, 'B' + testString, epoch, i)))

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                 os.path.join(OPATH,'images/{}/{}.nii'.format(
                                     self.date_time, 'tmp')))
        except: # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)

        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)


#===============================================================================
# Save and load

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join(PATH,'saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = os.path.join(PATH,'saved_models/{}/{}_weights_epoch_{}.hdf5'.format(self.date_time, model.name, epoch))
        model.save_weights(model_path_w)
        model_path_m = os.path.join(PATH,'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch))
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open(os.path.join(OPATH,'images/{}/loss_output.csv'.format(self.date_time)), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join(OPATH,'images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            'number of A test examples': len(self.A_test),
            'number of B test examples': len(self.B_test),
        })

        with open(os.path.join(OPATH,'images/{}/meta_data.json'.format(self.date_time)), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model, generate=False):
        if generate:
            path_to_model = os.path.join(OPATH,'generate_images', 'models', '{}.json'.format(model.name))
            path_to_weights = os.path.join(OPATH,'generate_images', 'models', '{}.hdf5'.format(model.name))
        else:
            join = os.path.join
            from os import listdir
            path = join(OPATH,'saved_models', '{}'.format(self.date_time))
            weight_files = [f for f in listdir(path) if f.endswith('hdf5') and model.name in f]
            epoch = []
            for f in weight_files:
                try:
                    epoch.append(int(f.replace('{}_weights_epoch_'.format(model.name),'').replace('.hdf5','')))
                except Exception as e:
                    print(e)
                    epoch.append(-1)
            epoch = np.asarray(epoch)
            resume_epoch = np.max(epoch)
            resume_weights = weight_files[np.argmax(epoch)]
            #path_to_model = os.path.join(OPATH,'images', 'models', '{}.json'.format(model.name))
            path_to_weights = os.path.join(path, resume_weights)
            self.initial_epoch = resume_epoch + 1
        #model = model_from_json(path_to_model)
        if True:
            model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self):
        response = input('Are you sure you want to generate synthetic images instead of training? (y/n): ')[0].lower()
        if response == 'y':
            self.load_model_and_weights(self.G_A2B)
            self.load_model_and_weights(self.G_B2A)
            synthetic_images_B = self.G_A2B.predict(self.A_test)
            synthetic_images_A = self.G_B2A.predict(self.B_test)

            def save_volume(image, name, domain):
                if self.channels == 1:
                    image = image[:, :, 0]
                #toimage(image, cmin=-1, cmax=1).save(os.path.join(
                #    'generate_images', 'synthetic_images', domain, name))
                img = nib.Nifti1Image(np.clip(image,-1,1),np.eye(4))
                nib.save(img,os.path.join(OPATH,'generate_images','synthetic_images',domain,name))

            # Test A images
            for i in range(len(synthetic_images_A)):
                # Get the name from the image it was conditioned on
                name = self.testB_image_names[i].strip('.nii') + '_synthetic.nii'
                synt_A = synthetic_images_A[i]
                save_volume(synt_A, name, 'A')

            # Test B images
            for i in range(len(synthetic_images_B)):
                # Get the name from the image it was conditioned on
                name = self.testA_image_names[i].strip('.nii') + '_synthetic.nii'
                synt_B = synthetic_images_B[i]
                save_volume(synt_B, name, 'B')

            print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
                  .format(len(self.A_test) + len(self.B_test)))


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 4:
                image = image[np.newaxis, :, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :, :]
                    self.images[random_id, :, :, :, :] = image[0, :, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images                                                   
        
n_blocks = 7
size = 128
unet = True

if not unet:
    # size of the latent space
    latent_dim = 100
    
tg = TubeProgCycleGAN()
G_A2B = tg.create_progressive_generator(name='G_A2B_model')
G_B2A = tg.create_progressive_generator(name='G_B2A_model')
g_models = [[g1,g2] for g1,g2 in zip(G_A2B,G_B2A)]
D_A2B = tg.create_progressive_discriminator(nblock=n_blocks,name='D_A2B_model')
D_B2A = tg.create_progressive_discriminator(nblock=n_blocks,name='D_B2S_model')
d_models = [[g1,g2] for g1,g2 in zip(D_A2B,D_B2A)]
gan_models = tg.create_progressive_composite()

sim_path = '/mnt/data2/retinasim/retinaGAN/trainA'
real_path = '/mnt/data2/retinasim/retinaGAN/trainB'
tg.opath = '/mnt/data2/retinasim/retina_cyclegan_output'

# load image data
print('loading images...')
nsample = 4 #512
real_images = load_real_samples(real_path,n=nsample)
sim_images = load_sim_samples(real_path,n=nsample)
print('Loaded: Real:{}, Sim:{}'.format(real_images.shape,sim_images.shape))

# train model
n_batch = [16, 16, 16, 8, 4, 4, 2]
#n_batch = [8, 8, 8, 4, 1, 1]
# 10 epochs == 500K images per training phase
#n_epochs = [5, 8, 8, 10, 10, 10]
n_epochs = [10, 32, 32, 250, 250, 250, 250]

tg.train(g_models, d_models, gan_models, real_images, sim_images, n_epochs, n_epochs, n_batch)  

import pdb
pdb.set_trace()



