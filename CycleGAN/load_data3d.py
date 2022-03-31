import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread
import nibabel as nib

def load_data(nr_of_channels, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0,
              path='data'):

    trainA_path = os.path.join(path, subfolder, 'trainA')
    trainB_path = os.path.join(path, subfolder, 'trainB')
    testA_path = os.path.join(path, subfolder, 'testA')
    testB_path = os.path.join(path, subfolder, 'testB')

    trainA_image_names = os.listdir(trainA_path)
    if nr_A_train_imgs != None:
        trainA_image_names = trainA_image_names[:nr_A_train_imgs]

    trainB_image_names = os.listdir(trainB_path)
    if nr_B_train_imgs != None:
        trainB_image_names = trainB_image_names[:nr_B_train_imgs]

    testA_image_names = os.listdir(testA_path)
    if nr_A_test_imgs != None:
        testA_image_names = testA_image_names[:nr_A_test_imgs]

    testB_image_names = os.listdir(testB_path)
    if nr_B_test_imgs != None:
        testB_image_names = testB_image_names[:nr_B_test_imgs]

    if generator:
        return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size, n_channel=nr_of_channels)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        trainA_images = create_image_array(trainA_image_names, trainA_path, nr_of_channels)
        trainB_images = create_image_array(trainB_image_names, trainB_path, nr_of_channels)
        testA_images = create_image_array(testA_image_names, testA_path, nr_of_channels)
        testB_images = create_image_array(testB_image_names, testB_path, nr_of_channels)
        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "testA_images": testA_images, "testB_images": testB_images,
                "trainA_image_names": trainA_image_names,
                "trainB_image_names": trainB_image_names,
                "testA_image_names": testA_image_names,
                "testB_image_names": testB_image_names}


def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if True: #image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                # NIFTI
                path = os.path.join(image_path, image_name)
                img = nib.load(path)
                image = np.array(img.dataobj)
                if image.shape[2]>8:                
                    #sums = np.sum(np.sum(image,axis=0),axis=0)
                    sums = np.max(np.max(image,axis=0),axis=0)

                    inds = np.linspace(0,image.shape[2]-1,image.shape[2],dtype='int')
                    thk = 8
                    #breakpoint()
                    #min_sum = image.shape[0]*image.shape[1]*20
                    min_sum = 128
                    sinds = np.where((sums>min_sum) & (inds>thk) & (inds<1500-thk))
                    #rnd = int(np.random.uniform(thk,1500-thk))
                    rnd = np.random.choice(sinds[0])
                    image = image[:,:,rnd:rnd+thk]
                
                #breakpoint()
                image = image[:, :, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                return None
 
            # Scale binary images
            #if 'trainA' in path: #np.max(array)==1 and np.min(array)==0:
            #    image[image>0] = 255
            #else:
            #    image *= 2 # for some reason tumour images only scale from 0-127...
                
            # Augment
            #print('augment')
            #Rotate
            nr = int(np.random.uniform(1,4))
            k = np.random.uniform(0,4,nr).astype('int')
            #axes = np.asarray([[0,1],[0,2],[1,2]])
            axes = np.asarray([[0,1]])
            axes_rnd = axes[np.random.choice(np.linspace(0,len(axes)-1,len(axes)),nr).astype('int')]
            for i in range(nr):
                image = np.rot90(image,k=k[i],axes=axes_rnd[i])
                
            # flip
            nr = int(np.random.uniform(1,4))
            flip_axes = [0,1] #  [0,1,2]
            axis = np.random.choice(flip_axes,nr)
            for i in range(nr):
                image = np.flip(image,axis=axis[i])

            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)
  
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
        
    # Scale from -1 to +1
    array = array / 127.5 - 1
    return array

class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1, n_channel=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if True: #image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if True: #image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:

        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, '', self.n_channel)
        real_images_B = create_image_array(batch_B, '', self.n_channel)

        return real_images_A, real_images_B  # input_data, target_data


if __name__ == '__main__':
    load_data()
