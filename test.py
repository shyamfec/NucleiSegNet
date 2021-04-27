import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import pandas as pd
from statistics import mean
import cv2
import warnings
import tensorflow
from tqdm import tqdm
from tensorflow.keras.callbacks import *
from color_norm import patch_join
from dataset import DataGeneratorFolder,aug_with_crop
from scores_comp import iou_metric,dice_metric
from model import create_model
warnings.filterwarnings('ignore')
# Path for the stain normalized image patches normalized image
test_data_path = '/content/TestData/'
# Path to the full sized test mask for score computation
gt_path = '/content/drive/MyDrive/intern_pyth/monuseg/TestData/Bin/'
# Path to model weight and weight name
model_path = '/content/drive/MyDrive/weights'
weight_name = 'weight_kumar_dataset.h5'
# Path to save the segmented masks
if not os.path.exists("/content/results"):
    os.mkdir("/content/results")
sv_path = '/content/results'

# used height and widht for patch
img_width_p = 256
img_height_p = 256
# Full image size
img_width_f = 1000
img_height_f = 1000

test_generator = DataGeneratorFolder(root_dir = test_data_path, 
                                    image_folder = 'tis/', 
                                    mask_folder = 'Bin/', 
                                    batch_size=1,augmentation = None,
                                    image_size=img_width_p,
                                    nb_y_features = 1)


model = create_model()
model.load_weights(os.path.join(model_path,weight_name))

out_im = []

print('Predicting the masks ===========>')
for tes in tqdm(range(len(test_generator)),total=len(test_generator)):
    Xtest_n, y_test_n  = test_generator.__getitem__(tes)
    predicted = model.predict(np.expand_dims(Xtest_n[0], axis=0)).reshape(img_width_p, img_height_p)
    predicted1= predicted.flatten()
    predicted1[predicted1>=0.5]=1
    predicted1[predicted1<0.5]=0
    predicted2 = predicted1.reshape((img_width_p, img_height_p))
    predicted2 = np.expand_dims(predicted2, -1)
    out_im.append(predicted2)

# Creating full sized segmented image (actual size) from segmented patches
print('Joining the segmented patches to original sized masks ===========>')
out_full,ids_test = patch_join(out_im)

# Writing the masks as image filee to folder
print('Writing segmented masks to image files ===========>')
for n, id_ in tqdm(enumerate(ids_test), total=len(ids_test)):

    imgs = np.reshape(out_full[n]*255,(img_width_f,img_height_f))
    filename = '{}/{}.png'.format(sv_path,os.path.splitext(id_)[0])
    cv2.imwrite(filename, imgs)

print('Segmented images are saved in {}'.format(sv_path))

# Computing scores (DICE and IOU)
print('Scores for the segmented output ===========>')
scr_met = {'IOU':[],'DICE':[]}

for _,i in enumerate(ids_test):
    
    gt = gt_path+os.path.splitext(i)[0]+'.png'
    plabel = os.path.join(sv_path,os.path.splitext(i)[0]+'.png')

    true   = cv2.imread(gt,0).astype(np.bool)
    pred_1 = cv2.imread(plabel,0).astype(np.bool)

    dice_coeff = dice_metric(true,pred_1)
    jacc_f = iou_metric(true,pred_1)

    scr_met['IOU'].append(jacc_f.item())
    scr_met['DICE'].append(dice_coeff.item())
    print('ID-{}  IOU: {:.3}, DICE: {:.3}'.format(os.path.splitext(id_)[0],jacc_f.item(),dice_coeff.item()))


print("mean of jaccard: ",mean(scr_met['IOU']))
print("mean of dice: ",mean(scr_met['DICE']))
    