from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D, concatenate, Add, merge
from keras_contrib.layers import InstanceNormalization
from keras.layers import InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
#from keras.engine.topology import Container
from keras.engine.network import Network as Container
from keras.constraints import max_norm

from collections import OrderedDict
#from scipy.misc import imsave, toimage  # has depricated
from keras.preprocessing.image import save_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.initializers import RandomNormal, Constant
from keras import backend
import numpy as np
arr = np.asarray
import random
import datetime
import time
import json
import math
import csv
import sys
import os
join = os.path.join

import keras.backend as K
import tensorflow as tf

import load_data
import nibabel as nib

PATH = '/mnt/data2/retinasim/retinaGAN2d' #'/mnt/ml/cycleGAN'
OPATH = '/mnt/data2/retinasim/retinaGAN2d' # '/home/simon/Desktop/Share/cycleGAN'

#PATH = '/mnt/data2/retinasim/retinaGAN2d/horse2zebra' #'/mnt/ml/cycleGAN'
#OPATH = '/mnt/data2/retinasim/retinaGAN2d/horse2zebra' # '/home/simon/Desktop/Share/cycleGAN'

LPATH = join(OPATH,'log')

# Set the logger
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


np.random.seed(seed=12345)

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


class ProgCycleGAN():
    def __init__(self, lr_D=2e-4, lr_G=2e-4, image_shape=(600,600,3), #(304, 256, 1),
                 date_time_string_addition='', image_folder='',date_time_string='', resume=False):
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

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = False

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
        #self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition
        
        self.create_full_model()
        prog_models = self.create_progressive_models(self.G_A2B,self.G_B2A,self.D_A.layers[1],self.D_B.layers[1])
        #breakpoint()
        prog_G_models = prog_models[0] + [self.G_model]
        prog_G_A2B = prog_models[1] + [self.G_A2B]
        prog_G_B2A = prog_models[2] + [self.G_B2A]
        prog_D_A = prog_models[3] + [self.D_A]
        prog_D_B = prog_models[4] + [self.D_B]
        prog_D_A_static = prog_models[5] + [self.D_A_static]
        prog_D_B_static = prog_models[6] + [self.D_B_static]
        image_shape = prog_models[7] + [[512,512,1]]

    # ======= Data ==========
        # Use 'None' to fetch all available images
        nr_A_train_imgs = None
        nr_B_train_imgs = None
        nr_A_test_imgs = None
        nr_B_test_imgs = None

        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if self.use_data_generator:
            self.data_generator = load_data.load_data(
                nr_of_channels=self.batch_size, generator=True, subfolder=image_folder, path=PATH) #, imsize=self.img_shape[0:2])

            # Only store test images
            nr_A_train_imgs = 0
            nr_B_train_imgs = 0

        data = load_data.load_data(nr_of_channels=self.channels,
                                   batch_size=self.batch_size,
                                   nr_A_train_imgs=nr_A_train_imgs,
                                   nr_B_train_imgs=nr_B_train_imgs,
                                   nr_A_test_imgs=nr_A_test_imgs,
                                   nr_B_test_imgs=nr_B_test_imgs,
                                   subfolder=image_folder,
                                   path=PATH) #,imsize=self.img_shape[0:2])

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.testA_image_names = data["testA_image_names"]
        self.testB_image_names = data["testB_image_names"]
        if not self.use_data_generator:
            print('Data has been loaded')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join(PATH,'images', self.date_time)

        if not os.path.exists(directory):
            os.makedirs(directory)
        #self.writeMetaDataToJSON()

        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
        
        self.initial_epoch = 0
        self.initial_imsize = 2
        if resume:         
            self.load_model_and_weights(self.G_A2B)
            self.load_model_and_weights(self.G_B2A)
            self.load_model_and_weights(self.D_A)
            self.load_model_and_weights(self.D_B)
            print('*** Resuming training at image size {} and epoch {} ***'.format(self.initial_imsize,self.initial_epoch))
        initial_epoch = self.initial_epoch
        initial_imsize = self.initial_imsize
       

        # ===== Tests ======
        # Simple Model
#         self.G_A2B = self.modelSimple('simple_T1_2_T2_model')
#         self.G_B2A = self.modelSimple('simple_T2_2_T1_model')
#         self.G_A2B.compile(optimizer=Adam(), loss='MAE')
#         self.G_B2A.compile(optimizer=Adam(), loss='MAE')
#         # self.trainSimpleModel()
#         self.load_model_and_generate_synthetic_images()

        # ======= Initialize training ==========
        sys.stdout.flush()
        #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        
        epochs = np.zeros(len(prog_G_models),dtype='int') + 200
        batch_size = np.ones(len(prog_G_models),dtype='int')
        fadein_fraction = np.zeros(len(prog_G_models),dtype='float') + 0.1
        decay_epoch_fraction = np.zeros(len(prog_G_models),dtype='float') + 1.
        im_size = [2,   4,   8,   16,  32,  64,  128,  256,  512]
        # Number of epochs
        ep =      [4,   50,  500, 500, 500, 500,  2000,2000, 2000] #
        epochs[:len(ep)] = ep
        # Batch size
        bs =      [10,  10,  5,   5,    5,   5,    1,    1,    1]
        batch_size[:len(bs)] = bs
        # Fade-in fraction
        #fr =      [0.25,0.25]
        #fadein_fraction[:len(fr)] = fr
        # Decay epoch fraction
        decay_epoch_fraction[im_size.index(16):] = 0.75
        
        m_init = im_size.index(initial_imsize)
        for i in range(m_init,len(prog_G_models)):
            self.G_model = prog_G_models[i]
            self.G_A2B = prog_G_A2B[i]
            self.G_B2A = prog_G_B2A[i]
            self.D_A = prog_D_A[i]
            self.D_B = prog_D_B[i]
            self.D_A_static = prog_D_A_static[i]
            self.D_B_static = prog_D_B_static[i]
            self.img_shape = image_shape[i] #[2,2,1]
            self.batch_size = batch_size[i]
            
            self.data_generator.batch_size = batch_size[i]
            self.data_generator.imsize = image_shape[i][0:2]
            
            data = load_data.load_data(nr_of_channels=self.channels,
                                   batch_size=batch_size[i],
                                   nr_A_train_imgs=nr_A_train_imgs,
                                   nr_B_train_imgs=nr_B_train_imgs,
                                   nr_A_test_imgs=nr_A_test_imgs,
                                   nr_B_test_imgs=nr_B_test_imgs,
                                   subfolder=image_folder,
                                   path=PATH,imsize=image_shape[i][0:2])

            self.A_train = data["trainA_images"]
            self.B_train = data["trainB_images"]
            self.A_test = data["testA_images"]
            self.B_test = data["testB_images"]
            self.testA_image_names = data["testA_image_names"]
            self.testB_image_names = data["testB_image_names"]
            
            self.decay_epoch = int(np.ceil(epochs[i]*fadein_fraction[i] + epochs[i]*(1-fadein_fraction[i])* decay_epoch_fraction[i]))  # The epoch where the linear decay of the learning rates start
            print('LR decay epoch: {}'.format(self.decay_epoch))
            
            self.train(epochs=epochs[i], batch_size=batch_size[i], save_interval=self.save_interval, imsize=image_shape[i], max_iter=-1,initial_epoch=initial_epoch,fadein_fraction=fadein_fraction[i])
            initial_epoch = 0
        #self.load_model_and_generate_synthetic_images()
        
    def weight_initialisation(self):
        return RandomNormal(stddev=0.02)
        
    def weight_constraint(self):
        return max_norm(1.0)

    def create_full_model(self):

        with tf.device("/gpu:1"):
            # optimizer
            self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
            self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

            # ======= Discriminator model ==========
            if self.use_multiscale_discriminator:
                D_A = self.modelMultiScaleDiscriminator()
                D_B = self.modelMultiScaleDiscriminator()
                loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
            else:
                D_A = self.modelDiscriminator()
                D_B = self.modelDiscriminator()
                loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
            # D_A.summary()

            # Discriminator builds
            image_A = Input(shape=self.img_shape)
            image_B = Input(shape=self.img_shape)
            guess_A = D_A(image_A)
            guess_B = D_B(image_B)
            self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
            self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

            # self.D_A.summary()
            # self.D_B.summary()
            self.D_A.compile(optimizer=self.opt_D,
                             loss=self.lse,
                             loss_weights=loss_weights_D)
            self.D_B.compile(optimizer=self.opt_D,
                             loss=self.lse,
                             loss_weights=loss_weights_D)

            # Use containers to avoid falsy keras error about weight descripancies
            self.D_A_static = Container(inputs=image_A, outputs=guess_A, name='D_A_static_model')
            self.D_B_static = Container(inputs=image_B, outputs=guess_B, name='D_B_static_model')

            # ======= Generator model ==========
            # Do note update discriminator weights during generator training
            self.D_A_static.trainable = False
            self.D_B_static.trainable = False

            # Generators
            self.G_A2B = self.modelGenerator(name='G_A2B_model')
            self.G_B2A = self.modelGenerator(name='G_B2A_model')
            # self.G_A2B.summary()

        if self.use_identity_learning:
            self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

        # Generator builds
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]

        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(self.lse)
                compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
                             
        # Embed layer sizes in the names
        for layer in self.G_A2B.layers:
            try:
                layer.name = layer.name + '_' + str(int(layer.output.shape[1]))
            except Exception as e:
                print(e)
                pass
        for layer in self.G_B2A.layers:
            try:
                layer.name = layer.name + '_' + str(int(layer.output.shape[1]))
            except Exception as e:
                print(e)
                pass
        # self.G_A2B.summary()
        
    # PROGRESSIVE MODELS ---------------------------

    def create_progressive_models(self, G_A2B,G_B2A,D_A,D_B,fadein=True,add_skips=True):
    
        depth0 = 0
    
        # Discriminators
        prog_D_A, prog_D_B, prog_D_A_static, prog_D_B_static, D_input_shape = [], [], [], [], []
        for j,model in enumerate([D_A,D_B]):
            
            nlayers = len(model.layers)
            conv_layers = [i for i,x in enumerate(model.layers) if 'conv2d' in x.name] #[:-1] #[:-2]
            depth = len(conv_layers)
            #breakpoint()
            
            for i in range(depth0+1,depth):
                enc_layers = model.layers[np.flip(conv_layers)[i]:]
                img_shape = [int(enc_layers[0].output.shape[1]),int(enc_layers[0].output.shape[2]),1]
                D_input_shape.append(img_shape)
                print(img_shape)
                input_layer = Input(shape=img_shape)
                # Linear scale-up
                #x = Conv2D(int(enc_layers[0].input.shape[-1]),(1,1),kernel_initializer=Constant(0.))(input_layer)
                x = Dense(int(enc_layers[0].input.shape[-1]),kernel_constraint=max_norm(1.0),kernel_initializer=RandomNormal(stddev=0.02))(input_layer) 
                #x.trainable = False
                nconv = [1 for x in enc_layers if 'conv2d' in x.name]
                conv_layers_cur, conv_inputs, ws_added = [], [], False
                for layer in enc_layers:
                    if 'conv2d' in layer.name:
                        conv_inputs.append(x)
                    if 'flatten' in layer.name:
                        x = Flatten()(x)
                        x = Dense(1)(x)
                        break
                    else:
                        print(x.name)
                        x = layer(x)
                    if 'conv2d' in layer.name:
                        conv_layers_cur.append(x)
                    elif 'leaky_re_lu'in layer.name:
                        if not ws_added and i>1 and fadein: # layer.name==model.layers[conv_layers[-1]].name and 
                            mp = MaxPooling2D(pool_size=(2, 2))(conv_inputs[-1])
                            if int(x.shape[-1])!=int(mp.shape[-1]):
                                # Linear scale-up
                                mp = Dense(int(x.shape[-1]),kernel_constraint=max_norm(1.0),kernel_initializer=RandomNormal(stddev=0.02))(mp)
                            x = WeightedSum()([mp, x])
                            ws_added = True
                    
                model1 = Model(input_layer,x)              

                # Use containers to avoid falsy keras error about weight descripancies
                image_A = Input(shape=img_shape)
                guess_A = model1(image_A)
                loss_weights_D = [0.5]
                
                if j==0:
                    name='D_A_model'
                else:
                    name = 'D_B_model'
                model1 = Model(inputs=image_A, outputs=guess_A, name=name)
                model1.compile(optimizer=self.opt_D,
                               loss=self.lse,
                               loss_weights=loss_weights_D)
                               
                if j==0:
                    prog_D_A.append(model1)
                    name = 'D_A_static_model_{}'.format(j)
                    prog_D_A_static.append(Container(inputs=image_A, outputs=guess_A, name=name))
                    prog_D_A_static[-1].trainable = False
                else:
                    prog_D_B.append(model1)
                    name = 'D_B_static_model_{}'.format(j)
                    prog_D_B_static.append(Container(inputs=image_A, outputs=guess_A, name=name))
                    prog_D_B_static[-1].trainable = False
        
        # Generators
        prog_G_A2B, prog_G_B2A, G_input_shape = [], [], []
        #print('Build prog Gen')
        for j,model in enumerate([G_A2B,G_B2A]):
        
            # Find  upsample layers
            nlayers = len(model.layers)
            transpose_layers = [i for i,x in enumerate(model.layers) if 'transpose' in x.name]
            conv_layers = [i for i,x in enumerate(model.layers) if 'conv' in x.name]
            nconv = len(transpose_layers)
            depth = nconv #int(nconv / 2)
            enc_conv_layers = np.asarray(([x for x in conv_layers if x not in transpose_layers and x<np.min(transpose_layers)]))
            
            for i in range(depth0,depth):
                
                # Identify encoding and decoding layers for current depth
                enc_dec_layers = model.layers[np.flip(enc_conv_layers)[i]:transpose_layers[i]+4] # add 3 to end to incorporate transpose layer, normalization and activations

                # Store image shape associated with current depth
                img_shape = [int(enc_dec_layers[0].output.shape[1]),int(enc_dec_layers[0].output.shape[2]),1]
                if j==0: # store layer dimensions
                    G_input_shape.append(img_shape)
                print(img_shape)
                
                # Create new input layer(s)
                if False:
                    input_layer = self.modelGenerator(input_only=True)
                    x = input_layer
                else:
                    #breakpoint()
                    input_layer = Input(shape=img_shape)
                    if img_shape[0]>16:
                        x = ReflectionPadding2D((3, 3))(input_layer)        
                    #else:
                    #x = self.c7Ak(x, 32)
                    x = input_layer
                
                # Linear scale-up
                if False:
                    x = Conv2D(int(enc_dec_layers[0].output.shape[-1]),(1,1))(x) #,kernel_initializer=Constant(0.))(x)
                elif True:
                    x = Dense(int(enc_dec_layers[0].output.shape[-1]),kernel_constraint=max_norm(1.0),kernel_initializer=RandomNormal(stddev=0.02))(x) #,kernel_initializer=Constant(0.))(x)
                    #x.trainable = False
                else:
                    pass
                #x.trainable = False
                
                #x = ReflectionPadding2D((3, 3))(x)
                
                scaleup_layer = x
                enc_dec_sizes, skips = [],[]
                ws1_added,ws2_added = False, False
                nc7Ak, ndk, nuk = 0, 0, 0
                tr_layers,tr_layer_input = [],[]
                for k,layer in enumerate(enc_dec_layers[:-1]): # Miss out final activation layer
                
                    #print(layer.name,x.shape)
                
                    # Store skip connections
                    if 'conv2d' in layer.name:
                        enc_dec_sizes.append(layer.get_output_at(-1))
                        if 'transpose' not in layer.name:
                            skips.append(x)
                            
                    # Add layers
                    if 'add' in layer.name:
                        if add_skips:
                            x = add([x, skips[-1]]) #concatenate([x, skips[-1]], axis=3)
                            skips.pop()
                    else:
                        #print(x.shape)
                        enc_block_in = x
                        if nc7Ak==0 and 'conv2d_' in layer.name:
                            #x = self.c7Ak(x, int(layer.output.shape[-1]))
                            print('c7Ak ',x.name,x.shape)
                            nc7Ak += 1
                        elif 'conv2d_' in layer.name and 'transpose' not in layer.name:
                            if False: # recreate
                                x,c_ = self.dk(x, int(layer.input.shape[-1])) # 64
                            else: # reuse
                                enc_block_size = 3
                                for kk in range(enc_block_size):
                                    x = enc_dec_layers[k+kk](x)
                                    print(x.name,x.shape)
                                    if 'activation' in x.name:
                                        break
                            #breakpoint()
                            ndk += 1
                        elif 'conv2d_transpose' in layer.name:
                            if nuk<ndk:
                                print(ndk,nuk)
                                tr_layers.append(layer)
                                tr_layer_input.append(x) #enc_dec_layers[k-1])
                                if False: # recreate
                                    nfilt = int(layer.input.shape[-1])
                                    x = self.uk(x, nfilt) # 64
                                else: # reuse
                                    enc_block_size = 5
                                    for kk in range(enc_block_size):
                                        if 'add' not in enc_dec_layers[k+kk].name:
                                            x = enc_dec_layers[k+kk](x)
                                            print(x.name,x.shape)
                                            if 'activation' in x.name:
                                                break
                                nuk += 1
                        elif False:
                            print(layer.name)
                            x = layer(x)
                        else:
                            pass
                        
                        # Add in weighted sum layer for blending
                        if fadein and 'conv2d' in layer.name and not ws1_added and ndk>0 and i>0:
                            mp = MaxPooling2D(pool_size=(2, 2))(scaleup_layer)
                            if int(x.shape[-1])!=int(mp.shape[-1]):
                                # Linear scale-up
                                #mp = Conv2D(int(x.shape[-1]),(1,1),kernel_initializer=Constant(0.))(mp)
                                mp = Dense(int(x.shape[-1]),kernel_constraint=max_norm(1.0),kernel_initializer=RandomNormal(stddev=0.02))(mp)
                                #mp.trainable = False
                                #print('Linear scale-up')
                            #breakpoint()
                            x = WeightedSum()([mp,x])
                            ws1_added = True

                if i>0 and fadein: # and False: # Add weighted sum layers to decoder for blending
                    #breakpoint()
                    mpd = UpSampling2D(size=(2, 2))(tr_layer_input[-1]) #.get_output_at(0))
                    if int(x.shape[-1])!=int(mpd.shape[-1]):
                        # Linear scale-down
                        #mpd = Conv2D(int(x.shape[-1]),(1,1),kernel_initializer=Constant(0.))(mpd)
                        mpd = Dense(int(x.shape[-1]),kernel_constraint=max_norm(1.0),kernel_initializer=RandomNormal(stddev=0.02))(mpd)
                        print(mpd.name)
                    x = WeightedSum()([mpd,x])
                    print(x.name)
                    
                    if False:
                        tr_layers = [y for y in enc_dec_layers if 'conv2d_transpose' in y.name]
                        #breakpoint()
                        # Identify second-to-last transpose layer
                        tr_2_layer_index = [i for i,y in enumerate(model.layers) if y.name==tr_layers[-2].name and y in enc_dec_layers]
                        # Offset transpose layer to activation layer
                        x_layer2 = model.layers[tr_2_layer_index[0]+3]
                        x = x_layer2.get_output_at(-1) 
                        #breakpoint()
                        mpd = UpSampling2D(size=(2, 2))(x) # Up-scale the second-to-last transpose layer's activation
                        #x = tr_layers[-1].get_output_at(-1) # Identify final transpose layer
                        tr_1_layer_index = [i for i,y in enumerate(model.layers) if y.name==tr_layers[-1].name and y in enc_dec_layers]
                        #x = tr_layers[-2].get_output_at(-1) 
                        # Offset transpose layer to activation layer
                        x_layer1 = model.layers[tr_1_layer_index[0]+3]
                        x = x_layer1.get_output_at(-1) 
                        #breakpoint()
                        if int(x.shape[-1])!=int(mpd.shape[-1]):
                            #breakpoint()
                            # Linear scale-down
                            #mpd = Conv2D(int(x.shape[-1]),(1,1),kernel_initializer=Constant(0.))(mpd)
                            mpd = Dense(int(x.shape[-1]),kernel_constraint=max_norm(1.0),kernel_initializer=RandomNormal(stddev=0.02))(mpd)
                            #x.trainable = False
                        #breakpoint()
                        x = WeightedSum()([mpd,x])
                    
                if False:
                    #init = self.weight_initialisation()
                    x = Conv2D(1, (1,1))(x) #,kernel_initializer=Constant(0.))(x)
                    #print(x.shape)
                    #x.trainable = False
                    x = Activation('tanh')(x)  # They say they use Relu but really they do not
                else:
                    #breakpoint()
                    if img_shape[0]>16:
                        x = ReflectionPadding2D((3, 3))(x)
                        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
                    else:
                        x = Conv2D(1, (1,1))(x)
                    x = Activation('tanh')(x)  # They say they use Relu but really they do not
                
                if j==0:
                    name = 'G_A2B_model'
                else:
                    name = 'G_B2A_model'    
                model1 = Model(input_layer,x,name=name)
                
                if j==0:
                    prog_G_A2B.append(model1)
                else:
                    prog_G_B2A.append(model1)
        
        # Generator builds
        nprog = len(prog_G_A2B)
        prog_G_model = []
        #breakpoint()
        for j in range(nprog):
            real_A = Input(shape=G_input_shape[j], name='real_A_{}'.format(j))
            real_B = Input(shape=G_input_shape[j], name='real_B_{}'.format(j))
            synthetic_B = prog_G_A2B[j](real_A)
            synthetic_A = prog_G_B2A[j](real_B)
            dA_guess_synthetic = prog_D_A_static[j](synthetic_A)
            dB_guess_synthetic = prog_D_B_static[j](synthetic_B)
            reconstructed_A = prog_G_A2B[j](synthetic_B)
            reconstructed_B = prog_G_B2A[j](synthetic_A)

            model_outputs = [reconstructed_A, reconstructed_B]
            compile_losses = [self.cycle_loss, self.cycle_loss,
                              self.lse, self.lse]
            compile_weights = [self.lambda_1, self.lambda_2,
                               self.lambda_D, self.lambda_D]

            if self.use_multiscale_discriminator:
                for _ in range(2):
                    compile_losses.append(self.lse)
                    compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
                for i in range(2):
                    model_outputs.append(dA_guess_synthetic[i])
                    model_outputs.append(dB_guess_synthetic[i])
            else:
                model_outputs.append(dA_guess_synthetic)
                model_outputs.append(dB_guess_synthetic)

            if self.use_supervised_learning:
                model_outputs.append(synthetic_A)
                model_outputs.append(synthetic_B)
                compile_losses.append('MAE')
                compile_losses.append('MAE')
                compile_weights.append(self.supervised_weight)
                compile_weights.append(self.supervised_weight)

            G_model = Model(inputs=[real_A, real_B],
                                 outputs=model_outputs,
                                 name='G_model')

            G_model.compile(optimizer=self.opt_G,
                                 loss=compile_losses,
                                 loss_weights=compile_weights)
                                 
            prog_G_model.append(G_model)
            
        return prog_G_model, prog_G_A2B, prog_G_B2A, prog_D_A, prog_D_B, prog_D_A_static, prog_D_B_static, G_input_shape

        
#===============================================================================
# Architecture functions

    def ck(self, x, k, use_normalization, stride):
        init = self.weight_initialisation()
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same',kernel_initializer=init)(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        init = self.weight_initialisation()
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid',kernel_initializer=init)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        init = self.weight_initialisation()
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same',kernel_initializer=init)(x)
        conv = x
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x, x

    def Rk(self, x0):
        init = self.weight_initialisation()
        k = int(x0.shape[-1])
        # first layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same',kernel_initializer=init)(x0)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same',kernel_initializer=init)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k, skip=None):
        init = self.weight_initialisation()
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid',kernel_initializer=init)(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same',kernel_initializer=init)(x)  # this matches fractionally stided with stride 1/2
            
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        if skip is not None:
            x = add([x, skip]) #concatenate([x, skip], axis=3)
        return x

#===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
    
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        
        ###TEST
        x = self.ck(x, 256, True, 2)
        x = self.ck(x, 256, True, 2)
        x = self.ck(x, 256, True, 2)
        x = self.ck(x, 256, True, 2)
        x = self.ck(x, 256, True, 2)
        
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        #x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None, input_only=False, add_skips=True):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)        
        x = self.c7Ak(x, 32)
        
        if input_only:
            return x
        
        # Layer 2
        x,c0 = self.dk(x, 64) # 256
        # Layer 3
        x,c1 = self.dk(x, 128) # 128
        
        ###TEST (512x512 image)
        x,c2 = self.dk(x, 256) # 64
        x,c3 = self.dk(x, 256) # 32
        x,c4 = self.dk(x, 256) # 16
        x,c5 = self.dk(x, 256) # 8
        x,c6 = self.dk(x, 256) # 4
        x,c7 = self.dk(x, 256) # 2
        #x,c8 = self.dk(x, 256) # Final one seems to break it...


        #if self.use_multiscale_discriminator:
        #    # Layer 3.5
        #    x,c9 = self.dk(x, 256)

        # Layer 4-12: Residual layer
        #for _ in range(11, 13):
        #    x = self.Rk(x)

        #if self.use_multiscale_discriminator:
        #    # Layer 12.5
        #    x = self.uk(x, 128)

        if add_skips:
            x = self.uk(x, 256,skip=c6) # 4
            x = self.uk(x, 256,skip=c5) # 8
            x = self.uk(x, 256,skip=c4) # 16
            x = self.uk(x, 256,skip=c3) # 32
            x = self.uk(x, 256,skip=c2) # 64
            x = self.uk(x, 128,skip=c1) # 128
            #x = self.uk(x, 256)
            # Layer 13
            x = self.uk(x, 64,skip=c0) # 256
        else:
            x = self.uk(x, 256)
            x = self.uk(x, 256)
            x = self.uk(x, 256)
            x = self.uk(x, 256)
            x = self.uk(x, 256)
            x = self.uk(x, 128)
            #x = self.uk(x, 256)
            # Layer 13
            x = self.uk(x, 64)
            
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

#===============================================================================
# Test - simple model
    def modelSimple(self, name=None):
        inputImg = Input(shape=self.img_shape)
        #x = Conv2D(1, kernel_size=5, strides=1, padding='same')(inputImg)
        #x = Dense(self.channels)(x)
        x = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputImg)
        x = Activation('relu')(x)
        x = Conv2D(self.channels, kernel_size=1, strides=1, padding='same')(x)

        return Model(input=inputImg, output=x, name=name)

    def trainSimpleModel(self):
        real_A = self.A_test[0]
        real_B = self.B_test[0]
        real_A = real_A[np.newaxis, :, :, :]
        real_B = real_B[np.newaxis, :, :, :]
        epochs = 200
        for epoch in range(epochs):
            print('Epoch {} started'.format(epoch))
            self.G_A2B.fit(x=self.A_train, y=self.B_train, epochs=1, batch_size=1)
            self.G_B2A.fit(x=self.B_train, y=self.A_train, epochs=1, batch_size=1)
            #loss = self.G_A2B.train_on_batch(x=real_A, y=real_B)
            #print('loss: ', loss)
            synthetic_image_A = self.G_B2A.predict(real_B, batch_size=1)
            synthetic_image_B = self.G_A2B.predict(real_A, batch_size=1)
            self.save_tmp_images(real_A, real_B, synthetic_image_A, synthetic_image_B)

        self.saveModel(self.G_A2B, 200)
        self.saveModel(self.G_B2A, 200)

#===============================================================================
# Training
    def train(self, epochs, batch_size=1, save_interval=1, imsize=None, max_iter=-1, initial_epoch=1, fadein_fraction=0.25):
        def run_training_iteration(loop_index, epoch_iterations, fadein=True, fadein_steps=10, counter=1):
        
            if fadein:
                alpha = self.update_fadein([self.G_A2B, self.G_B2A, self.D_A, self.D_B], counter, fadein_steps)
        
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
            print('Size-----------------', imsize)
            print('Batch size-----------', batch_size)
            print('Alpha----------------', alpha)
            print('Loop index-----------', loop_index + 1, '/', epoch_iterations)
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
        
        counter = 1

        for epoch in range(initial_epoch, epochs + 1):

            if self.use_data_generator:
                loop_index = 1
                
                for ii in range(len(self.data_generator)):
                    images = self.data_generator[ii]
                    if ii==max_iter:
                        break
                    real_images_A = images[0]
                    real_images_B = images[1]
                    if real_images_A.shape[0]!=batch_size or real_images_B.shape[0]!=batch_size:
                        breakpoint()
                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]

                    # Run all training steps
                    #fadein_fraction = 0.25
                    fadein_steps = int(np.ceil(epochs*len(self.data_generator)*fadein_fraction))
                    run_training_iteration(loop_index, self.data_generator.__len__(),fadein_steps=fadein_steps,counter=counter)

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break

                    loop_index += 1
                    counter += 1

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
                    run_training_iteration(loop_index, epoch_iterations)

            #================== within epoch loop end ==========================

            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                print('2D version!"')
                self.saveImages(epoch, imsize[0], real_images_A, real_images_B)

            if epoch % 5 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_A, epoch, imsize[0])
                self.saveModel(self.D_B, epoch, imsize[0])
                self.saveModel(self.G_A2B, epoch, imsize[0])
                self.saveModel(self.G_B2A, epoch, imsize[0])

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

    # update the alpha value on each instance of WeightedSum
    def update_fadein(self, models, step, n_steps, alpha=None):
        # calculate current alpha (linear from 0 to 1). Alpha=0 weights first input var; Alpha=1 weights second input var
        if alpha is None:
            if n_steps<=1:
                alpha = 1
            else:
                alpha = step / float(n_steps - 1)
                if alpha>1:
                    alpha = 1.
        # update the alpha for each model
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    backend.set_value(layer.alpha, alpha)
                    
        return alpha

#===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0].squeeze()

        #toimage(image, cmin=-1, cmax=1).save(path_name)
        #save_img(path_name, image)
        from skimage.transform import resize
        dim = 512
        #image = resize(image, [dim*2,dim*3])
        img = nib.Nifti1Image(np.clip(image,-1,1),np.eye(4))
        nib.save(img,path_name)

    def saveImages(self, epoch, imsize, real_image_A, real_image_B, num_saved_images=1):
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

            synthetic_image_B = self.G_A2B.predict(real_image_A)
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
                                 join(OPATH,'images/{}/{}/size{}_epoch{}_sample{}.nii'.format(
                                     self.date_time, 'A' + testString, imsize, epoch, i)))
            self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
                                 join(OPATH,'images/{}/{}/size{}_epoch{}_sample{}.nii'.format(
                                     self.date_time, 'B' + testString, imsize, epoch, i)))

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                 join(OPATH,'images/{}/{}.nii'.format(
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

    def saveModel(self, model, epoch, imsize):
        # Create folder to save model architecture and weights
        directory = os.path.join(OPATH,'saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = join(OPATH,'saved_models/{}/{}_weights_imsize_{}_epoch_{}.hdf5'.format(self.date_time, model.name, imsize, epoch))
        model.save_weights(model_path_w)
        model_path_m = join(OPATH,'saved_models/{}/{}_model_imsize_{}_epoch_{}.json'.format(self.date_time, model.name, imsize, epoch))
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open(join(OPATH,'images/{}/loss_output.csv'.format(self.date_time)), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join('images', self.date_time)
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

        with open(PATH+'images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
    
        from os import listdir
        path = join(OPATH,'saved_models', '{}'.format(self.date_time))
        weight_files = [f for f in listdir(path) if f.endswith('hdf5') and model.name in f]
        ## TEMP FIX
        if len(weight_files)==0:
            weight_files = [f for f in listdir(path) if f.endswith('hdf5') and model.name.replace('_model','') in f]
        ###########
        epoch, imsize = [], []
        for f in weight_files:
            try:
                tmp = f.split('_')
                imsize.append(int(tmp[tmp.index('imsize')+1]))
                epoch.append(int(tmp[tmp.index('epoch')+1].replace('.hdf5','')))
            except Exception as e:
                print(e)
                epoch.append(-1)
                imsize.append(-1)

        epoch = np.asarray(epoch)
        imsize = np.asarray(imsize)
        resume_size = np.max(imsize)
        inds = np.where(imsize==resume_size)
        resume_epoch = np.max(epoch[inds])
        inds = np.where((imsize==resume_size) & (epoch==resume_epoch))
        #resume_weights = weight_files[np.argmax(epoch)]
        resume_weights = weight_files[inds[0][0]]
        #path_to_model = os.path.join(OPATH,'images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join(path, resume_weights)
        self.initial_epoch = resume_epoch + 1
        self.initial_imsize = resume_size
    
        #path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        #path_to_weights = os.path.join('generate_images', 'models', '{}.hdf5'.format(model.name))
        ##model = model_from_json(path_to_model)
        #model.load_weights(path_to_weights)       

    def load_model_and_generate_synthetic_images(self):
        response = input('Are you sure you want to generate synthetic images instead of training? (y/n): ')[0].lower()
        if response == 'y':
            self.load_model_and_weights(self.G_A2B)
            self.load_model_and_weights(self.G_B2A)
            synthetic_images_B = self.G_A2B.predict(self.A_test)
            synthetic_images_A = self.G_B2A.predict(self.B_test)

            def save_image(image, name, domain):
                if self.channels == 1:
                    image = image[:, :, 0]
                #toimage(image, cmin=-1, cmax=1).save(os.path.join(
                #    'generate_images', 'synthetic_images', domain, name))
                save_img(os.path.join('generate_images','synthetic_images',domain,name), image)

            # Test A images
            for i in range(len(synthetic_images_A)):
                # Get the name from the image it was conditioned on
                name = self.testB_image_names[i].strip('.png') + '_synthetic.png'
                synt_A = synthetic_images_A[i]
                save_image(synt_A, name, 'A')

            # Test B images
            for i in range(len(synthetic_images_B)):
                # Get the name from the image it was conditioned on
                name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
                synt_B = synthetic_images_B[i]
                save_image(synt_B, name, 'B')

            print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
                  .format(len(self.A_test) + len(self.B_test)))


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


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
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

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
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
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


if __name__ == '__main__':

    GAN = ProgCycleGAN(image_shape=[512,512,1],date_time_string='20220408-092046',resume=True)
