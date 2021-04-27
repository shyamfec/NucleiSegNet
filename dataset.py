from skimage.io import imread
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

# tf.logging.set_verbosity(tf.logging.ERROR)
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline
from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
 
import albumentations as albu
from albumentations import Resize, Crop
# from  albumentations.augmentations.transforms import GaussianBlur

def aug_with_crop(image_size = 256, crop_prob = 1):
    return Compose([
        RandomCrop(width = image_size, height = image_size, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        # ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        # RandomGamma(p=0.25),
        # IAAEmboss(p=0.25),
        Blur(p=0.3, blur_limit = 3),
        # GaussianBlur(p=0.5, blur_limit = 3),
        # OneOf([
        #     ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     GridDistortion(p=0.5),
        #     OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
        # ], p=0.5)
    ], p = 1)


class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir=r'/data/val_test', image_folder='tis/', mask_folder='Bin/', 
                 batch_size=1, image_size=256, nb_y_features=1, 
                 augmentation=None,
                 suffle=True):
        self.image_filenames = sorted(os.listdir(os.path.join(root_dir, image_folder)))
        self.mask_names = sorted(os.listdir(os.path.join(root_dir, mask_folder)))
        self.batch_size = batch_size
        self.currentIndex = 0
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.suffle = suffle
        self.path1= root_dir
    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.suffle==True:
            self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)
        

    def read_image_mask(self, image_name, mask_name,path1):
        i_path=path1+"/tis/"
        m_path = path1+"/Bin/"

        img1 = cv2.imread(i_path+image_name)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        img2 = img2/255
        mask1 = cv2.imread(m_path+mask_name,0)
        mask1 = mask1.astype(np.uint8)

        return img2, mask1

    def __getitem__(self, index):
        """
        Generate one batch of data
        
        """

        # Generate indexes of the batch
        data_index_min = int(index*self.batch_size)

        # data_index_max = int(min((index+1)*self.batch_size, len(self.image_filenames)))
        data_index_max = int((index+1)*self.batch_size)
        indexes = self.image_filenames[data_index_min:data_index_max]
        
        this_batch_size = len(indexes) # The last batch can be smaller than the others

        # Defining dataset
        X = np.zeros((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.zeros((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.bool)

        for i, sample_index in enumerate(indexes):
            # print(sample_index)
            # print(index)
            X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i], 
                                                    self.mask_names[index * self.batch_size + i],self.path1)
  
            # if augmentation is defined, we assume its a train set
            if self.augmentation is not None:
                  
                # Augmentation code
                augmented = self.augmentation(self.image_size)(image=X_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)
                X[i, ...] = np.clip(image_augm, a_min = 0, a_max=1)
                y[i, ...] = mask_augm
            
            # if augmentation isnt defined, we assume its a test set. 
            # Because test images can have different sizes we resize it to be divisable by 32
            elif self.augmentation is None and self.batch_size ==1:
                X_sample, y_sample = self.read_image_mask(self.image_filenames[index * 1 + i], 
                                                      self.mask_names[index * 1 + i],self.path1)
                # augmented = Resize(height=(X_sample.shape[0]//32)*32, width=(X_sample.shape[1]//32)*32)(image = X_sample, mask = y_sample)
                augmented = RandomCrop(width = self.image_size, height = self.image_size, p=1)(image = X_sample, mask = y_sample)
                X_sample, y_sample = augmented['image'], augmented['mask']
                
                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32),\
                       y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.bool)

        return X, y