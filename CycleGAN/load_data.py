import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread
join = os.path.join
import nibabel as nib
from matplotlib import pyplot as plt

def load_data(nr_of_channels, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',path='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0,
              imsize=None):

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
        return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size, imsize=imsize)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        trainA_images = create_image_array(trainA_image_names, trainA_path, nr_of_channels, imsize=imsize)
        trainB_images = create_image_array(trainB_image_names, trainB_path, nr_of_channels, imsize=imsize)
        testA_images = create_image_array(testA_image_names, testA_path, nr_of_channels, imsize=imsize)
        testB_images = create_image_array(testB_image_names, testB_path, nr_of_channels, imsize=imsize)
        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "testA_images": testA_images, "testB_images": testB_images,
                "trainA_image_names": trainA_image_names,
                "trainB_image_names": trainB_image_names,
                "testA_image_names": testA_image_names,
                "testB_image_names": testB_image_names}


def create_image_array(image_list, image_path, nr_of_channels, imsize=[512,512]):

    image_array = []
    for image_name in image_list:
        if True: #image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                path = os.path.join(image_path, image_name)
                if image_name.lower().endswith('.nii'):
                    # NIFTI
                    img = nib.load(path)
                    image = np.array(img.dataobj)
                    augment = True
                    if 'trainB' in path and image.max()<=1.:
                         #breakpoint()
                         image *= 256
                    
                elif image_name.lower().endswith('.jpg'):
                    image = np.array(Image.open(path).convert('L'))
                    image = np.rot90(image,k=1,axes=[0,1])
                    augment = False
                
                if imsize is not None:
                    #print('imsize:',imsize)
                    from skimage.transform import resize
                    image = resize(image, imsize) * 255
                
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                return None
 
            if augment:
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


    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)
  
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
    return array


class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1, imsize=None):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        self.imsize = imsize
        for image_name in image_list_A:
            if True: #image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if True: #image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if False: #idx >= min(len(self.train_A), len(self.train_B)):
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
            x0,x1 = idx * self.batch_size, (idx + 1) * self.batch_size
            if x0>=len(self.train_A):
                inds = np.random.choice(np.linspace(0,len(self.train_A)-1,len(self.train_A),dtype='int'),self.batch_size)
                batch_A = [self.train_A[x] for x in inds]
            elif x1>=len(self.train_A):
                inds = np.arange(x0,len(self.train_A)-1,1,dtype='int')
                inds = np.concatenate([inds,np.random.choice(np.linspace(0,len(self.train_A)-1,len(self.train_A),dtype='int'),self.batch_size-inds.shape[0])])
                batch_A = [self.train_A[x] for x in inds]
            else:
                batch_A = self.train_A[x0:x1]
                
            x0,x1 = idx * self.batch_size, (idx + 1) * self.batch_size
            if x0>=len(self.train_B): # If outside range of images, draw randomly
                inds = np.random.choice(np.linspace(0,len(self.train_B)-1,len(self.train_B),dtype='int'),self.batch_size)
                batch_B = [self.train_B[x] for x in inds]
            elif x1>=len(self.train_B): # If end limit is outside image range, add on random selection to end
                inds = np.linspace(x0,len(self.train_B)-1,len(self.train_B)-x0,dtype='int')
                inds = np.concatenate([inds,np.random.choice(np.linspace(0,len(self.train_B)-1,len(self.train_B),dtype='int'),self.batch_size-inds.shape[0])])
                batch_B = [self.train_B[x] for x in inds]
            else:
                batch_B = self.train_B[x0:x1]
            #batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        #if imsize is None:
        imsize = self.imsize
        real_images_A = create_image_array(batch_A, '', 1, imsize=imsize)
        real_images_B = create_image_array(batch_B, '', 1, imsize=imsize)

        return real_images_A, real_images_B  # input_data, target_data


if __name__ == '__main__':
    load_data()
