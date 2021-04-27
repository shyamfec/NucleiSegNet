import numpy as np
import matplotlib.pyplot as plt
import spams
import cv2

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
from pathlib import Path
import os
import sys
import random
import warnings
import pandas as pd

from tqdm import tqdm
from itertools import chain
import math
from vahadane import vahadane

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
M_CHANNEL=1
Res_HEIGHT = 1000  # actual image height
Res_WIDTH = 1000   # actual image width
#no of patches = (input image size/ crop size)^2  per image .
pat = 16  
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
np.random.seed = seed
# path where you want to store the stain normalized images
#----- # Test # ------#
Path("/content/TestData").mkdir(parents=True, exist_ok=True)
Path("/content/TestData/Bin").mkdir(parents=True, exist_ok=True) # for masks
Path("/content/TestData/tis").mkdir(parents=True, exist_ok=True) # for tissues

bin_p_ts = '/content/TestData/Bin'
tis_p_ts = '/content/TestData/tis'
#----- # Train # ------#
Path("/content/TrainData").mkdir(parents=True, exist_ok=True)
Path("/content/TrainData/Bin").mkdir(parents=True, exist_ok=True) # for masks
Path("/content/TrainData/tis").mkdir(parents=True, exist_ok=True) # for tissues

bin_p_tr = '/content/TrainData/Bin/'
tis_p_tr = '/content/TrainData/tis/'
#----- # Valid # ------#
Path("/content/ValidData").mkdir(parents=True, exist_ok=True)
Path("/content/ValidData/Bin").mkdir(parents=True, exist_ok=True) # for masks
Path("/content/ValidData/tis").mkdir(parents=True, exist_ok=True) # for tissues

bin_p_vl = '/content/ValidData/Bin/'
tis_p_vl = '/content/ValidData/tis/'

# Give path to your dataset
Train_image_path = '/content/drive/MyDrive/intern_pyth/monuseg/TrainData/original_images/'
Train_mask_path = '/content/drive/MyDrive/intern_pyth/monuseg/TrainData/Bin/'

val_image_path = '/content/drive/MyDrive/intern_pyth/monuseg/ValidData/original_images/'
val_mask_path = '/content/drive/MyDrive/intern_pyth/monuseg/ValidData/Bin/'

Test_image_path = '/content/drive/MyDrive/intern_pyth/monuseg/TestData/tis/'
test_mask_path = '/content/drive/MyDrive/intern_pyth/monuseg/TestData/Bin/'

# Give a reference image path for stain normalization
reference_image = '/content/drive/MyDrive/intern_pyth/monuseg/TestData/tis/TCGA-21-5784-01Z-00-DX1.tif'


# getting the train and test ids
train_ids1 = next(os.walk(Train_image_path))[2]
train_mask_ids1 = next(os.walk(Train_mask_path))[2]
val_ids1 = next(os.walk(val_image_path))[2]
val_mask_ids1 = next(os.walk(val_mask_path))[2]
test_ids1 = next(os.walk(Test_image_path))[2]
test_mask_ids1 = next(os.walk(test_mask_path))[2]

# sorting the train and test ids
train_ids = sorted(train_ids1,key=lambda x: (os.path.splitext(x)[0]))
train_mask_ids = sorted(train_mask_ids1,key=lambda x: (os.path.splitext(x)[0]))
test_ids = sorted(test_ids1,key=lambda x: (os.path.splitext(x)[0]))
test_mask_ids = sorted(test_mask_ids1,key=lambda x: (os.path.splitext(x)[0]))
val_ids = sorted(val_ids1,key=lambda x: (os.path.splitext(x)[0]))
val_mask_ids = sorted(val_mask_ids1,key=lambda x: (os.path.splitext(x)[0]))

def stain_norm_patch():

    def read_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
        p = np.percentile(img, 90)
        img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
        return img

    def vaha(SOURCE_PATH,TARGET_PATH):
        source_image = read_image(SOURCE_PATH)
        target_image = read_image(TARGET_PATH)
        vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
        # vhd.show_config()

        Ws, Hs = vhd.stain_separate(source_image)
        vhd.fast_mode=0;vhd.getH_mode=0;
        Wt, Ht = vhd.stain_separate(target_image)
        img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
        return img

    def rein(src):
        # stain_normalizer 'Vahadane'
        target_img = reference_image
        im_nmzd = vaha(src,target_img)
        return im_nmzd

    # Get and resize train images and masks
    def train():
        X_train = np.zeros((len(train_ids)*pat, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        Y_train = np.zeros((len(train_ids)*pat, IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
        print('stain normalizing and cropping patches of train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

            img = rein(Train_image_path + id_)   
            mask_ = cv2.imread(Train_mask_path + (os.path.splitext(id_)[0])+'.png',0)
            mask_ = np.expand_dims(mask_, -1)

            temp_list = []
            temp_list_mask = []
            for i in range (int(math.pow(pat,0.5))):
                for j in range(int(math.pow(pat,0.5))):
                    if i<(int(math.pow(pat,0.5))-1):
                        if j<(int(math.pow(pat,0.5))-1):
                            crop_img1 = img[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            crop_mask1 = mask_[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            temp_list.append(crop_img1)
                            temp_list_mask.append(crop_mask1)
                        elif j==(int(math.pow(pat,0.5))-1):
                            crop_img2 = img[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            crop_mask2 = mask_[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            temp_list.append(crop_img2)
                            temp_list_mask.append(crop_mask2)
                    elif i==(int(math.pow(pat,0.5))-1):
                        if j<(int(math.pow(pat,0.5))-1):
                            crop_img3 = img[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            crop_mask3 = mask_[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            temp_list.append(crop_img3)
                            temp_list_mask.append(crop_mask3)
                        elif j==(int(math.pow(pat,0.5))-1):
                            crop_img4 = img[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            crop_mask4 = mask_[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            temp_list.append(crop_img4)
                            temp_list_mask.append(crop_mask4)

            for t in range(0,pat):
                X_train[n*pat+t] = temp_list[t]
                Y_train[n*pat+t] = temp_list_mask[t]    
                # mask = np.maximum(mask, mask_)
        return X_train, Y_train


    def val():
        X_val = np.zeros((len(val_ids)*pat, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        Y_val = np.zeros((len(val_ids)*pat, IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
        print('stain normalizing and cropping patches of validation images and masks ... ')
        sys.stdout.flush()
        for m, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):

            val_img = rein(val_image_path + id_)   
            val_mask_ = cv2.imread(val_mask_path + (os.path.splitext(id_)[0])+'.png',0)
            val_mask_ = np.expand_dims(val_mask_, -1)

            temp_list = []
            temp_list_mask = []
            for i in range (int(math.pow(pat,0.5))):
                for j in range(int(math.pow(pat,0.5))):
                    if i<(int(math.pow(pat,0.5))-1):
                        if j<(int(math.pow(pat,0.5))-1):
                            crop_val_img1 = val_img[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            crop_mask1 = val_mask_[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            temp_list.append(crop_val_img1)
                            temp_list_mask.append(crop_mask1)
                        elif j==(int(math.pow(pat,0.5))-1):
                            crop_val_img2 = val_img[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            crop_mask2 = val_mask_[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            temp_list.append(crop_val_img2)
                            temp_list_mask.append(crop_mask2)
                    elif i==(int(math.pow(pat,0.5))-1):
                        if j<(int(math.pow(pat,0.5))-1):
                            crop_val_img3 = val_img[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            crop_mask3 = val_mask_[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            temp_list.append(crop_val_img3)
                            temp_list_mask.append(crop_mask3)
                        elif j==(int(math.pow(pat,0.5))-1):
                            crop_val_img4 = val_img[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            crop_mask4 = val_mask_[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            temp_list.append(crop_val_img4)
                            temp_list_mask.append(crop_mask4)

            for t in range(0,pat):
                X_val[m*pat+t] = temp_list[t]
                Y_val[m*pat+t] = temp_list_mask[t]    
                # mask = np.maximum(mask, mask_)
        return X_val, Y_val


    def test():
        X_test = np.zeros((len(test_ids)*pat, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        Y_test = np.zeros((len(test_ids)*pat, IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
        print('stain normalizing and cropping patches of test images ... ')
        sys.stdout.flush()
        
        for s, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

            img = rein(Test_image_path + id_)   
            test_mask_ = cv2.imread(test_mask_path + (os.path.splitext(id_)[0])+'.png',0)
            test_mask_ = np.expand_dims(test_mask_, -1)

            temp_list = []
            temp_list_mask = []
            for i in range (int(math.pow(pat,0.5))):
                for j in range(int(math.pow(pat,0.5))):
                    if i<(int(math.pow(pat,0.5))-1):
                        if j<(int(math.pow(pat,0.5))-1):
                            crop_img1 = img[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            crop_mask1 = test_mask_[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            temp_list.append(crop_img1)
                            temp_list_mask.append(crop_mask1)
                        elif j==(int(math.pow(pat,0.5))-1):
                            crop_img2 = img[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            crop_mask2 = test_mask_[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            temp_list.append(crop_img2)
                            temp_list_mask.append(crop_mask2)
                    elif i==(int(math.pow(pat,0.5))-1):
                        if j<(int(math.pow(pat,0.5))-1):
                            crop_img3 = img[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            crop_mask3 = test_mask_[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH]
                            temp_list.append(crop_img3)
                            temp_list_mask.append(crop_mask3)
                        elif j==(int(math.pow(pat,0.5))-1):
                            crop_img4 = img[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            crop_mask4 = test_mask_[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24]
                            temp_list.append(crop_img4)
                            temp_list_mask.append(crop_mask4)

            for t in range(0,pat):
                X_test[s*pat+t] = temp_list[t]
                Y_test[s*pat+t] = temp_list_mask[t]    
                # mask = np.maximum(mask, mask_)
        return X_test, Y_test

    train1 = train()
    X_train = train1[0]
    Y_train = train1[1]

    val1 = val()
    X_val = val1[0]
    Y_val = val1[1]

    test1 = test()
    X_test = test1[0]
    Y_test = test1[1]

    # this will save the stain normalized patches into the created paths above     
    #------------------------#TEST#---------------------------------#

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        id_1 = os.path.splitext(id_)[0]
        for j in range(16):
            j1 = "{0:0=2d}".format(j)
            img_t = X_test[n*16+j]
            imgs_b = Y_test[n*16+j]*255    
            # img_t = X_test[n]
            # imgs_b = np.reshape(Y_test[n]*255,(IMG_WIDTH,IMG_HEIGHT))
            filename1 = '{}/{}_{}.png'.format(tis_p_ts,id_1,j1)
            cv2.imwrite(filename1, cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
            filename2 = '{}/{}_{}.png'.format(bin_p_ts,id_1,j1)
            cv2.imwrite(filename2, imgs_b)
    #------------------------#VAL#-------------------------------#

    for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):
        id_1 = os.path.splitext(id_)[0]

        for j in range(16):
            j1 = "{0:0=2d}".format(j)
            img_t = X_val[n*16+j]
            imgs_b = Y_val[n*16+j]*255   
            filename1 = '{}/{}_{}.png'.format(tis_p_vl,id_1,j1)
            cv2.imwrite(filename1,cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))    #cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
            filename2 = '{}/{}_{}.png'.format(bin_p_vl,id_1,j1)
            cv2.imwrite(filename2, imgs_b)
    #------------------------#TRAIN#-------------------------------#

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        id_1 = os.path.splitext(id_)[0]

        for j in range(16):
            j1 = "{0:0=2d}".format(j)
            img_t = X_train[n*16+j]
            imgs_b = Y_train[n*16+j]*255   
            filename1 = '{}/{}_{}.png'.format(tis_p_tr,id_1,j1)
            cv2.imwrite(filename1, cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))  #cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
            filename2 = '{}/{}_{}.png'.format(bin_p_tr,id_1,j1)
            cv2.imwrite(filename2, imgs_b)
        
def patch_join(out_im):
    num_im = len(out_im)//pat
    num_pat = int(pat**0.5)
    out_concat = np.zeros((Res_HEIGHT, Res_WIDTH, 1), dtype=np.uint8)
    # Y_concat = np.zeros((Res_HEIGHT, Res_WIDTH, 1), dtype=np.bool)

    out_full = np.zeros((num_im,Res_HEIGHT, Res_WIDTH, 1), dtype=np.uint8)
    # Y_full = np.zeros((num_im,Res_HEIGHT, Res_WIDTH, 1), dtype=np.bool)


    for k in range(num_im):
        sec1 = []
        y_sec1 = []
        for l in range(pat):
        
            sec = out_im[k*pat+l]
            sec1.append(sec)

        for i in range(int(num_pat)):
            for j in range(int(num_pat)): 
                
                if i<num_pat-1:
                    if j<num_pat-1:
                        out_concat[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH] = sec1[i*num_pat+j]

                    elif j==num_pat-1:
                        out_concat[i*IMG_HEIGHT:i*IMG_HEIGHT+IMG_HEIGHT, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24] = sec1[i*num_pat+j]

                elif i==num_pat-1:
                    if j<num_pat-1:
                        out_concat[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH:j*IMG_WIDTH+IMG_WIDTH] = sec1[i*num_pat+j]

                    elif j==num_pat-1:
                        out_concat[i*IMG_HEIGHT-24:i*IMG_HEIGHT+IMG_HEIGHT-24, j*IMG_WIDTH-24:j*IMG_WIDTH+IMG_WIDTH-24] = sec1[i*num_pat+j]
         
        out_full[k] = out_concat
    
    return out_full,test_ids

if __name__ == '__main__':
    stain_norm_patch()