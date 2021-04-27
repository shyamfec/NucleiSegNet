import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import pandas as pd
import warnings
import tensorflow
from tensorflow.keras.callbacks import *
from dataset import DataGeneratorFolder,aug_with_crop
from model import create_model



# Path to the stain normalized patch 
valid_data_path = '/content/ValidData/'
train_data_path = '/content/TrainData/'

# path for the weight and the saved history scores( loss and metrices) over the epochs
if not os.path.exists("/content/checkpoint"):
    os.mkdir("/content/checkpoint")

if not os.path.exists("/content/history"):
    os.mkdir("/content/history")


model_path = '/content/checkpoint' 
weight_name = 'nuclei_seg.h5'  # name of the weight for the final model
history_path = '/content/history'
hist_name = 'nuclei_seg.csv'  # name of the csv file for storing all scores

patch_size = 256
bs = 4 # batch size for training for validation it is taken 1
eps = 40 # epochs

warnings.filterwarnings('ignore')

val_generator = DataGeneratorFolder(root_dir = valid_data_path, 
                                    image_folder = 'tis/', 
                                    mask_folder = 'Bin/', 
                                    batch_size=1,augmentation = None,
                                    image_size=patch_size,
                                    nb_y_features = 1)

train_generator = DataGeneratorFolder(root_dir = train_data_path, 
                                      image_folder = 'tis/', 
                                      mask_folder = 'Bin/', 
                                      augmentation = aug_with_crop,
                                      batch_size=bs,
                                      image_size=patch_size,
                                      nb_y_features = 1)


# reduces learning rate on plateau
lr_reducer = ReduceLROnPlateau(factor=0.1,patience=5,
                               cooldown= 5,
                               min_lr=0.1e-5,verbose=1)
# # model autosave callbacks
# mode_autosave = ModelCheckpoint("kidney_mod_kumar.h5", 
#                                 monitor='val_f1-score', 
#                                 mode='max', save_best_only=True, verbose=1, save_freq=65)

# # stop learining as metric on validatopn stop increasing
# early_stopping = EarlyStopping(patience=5, verbose=1, mode = 'auto') 

# # tensorboard for monitoring logs
# tensorboard = TensorBoard(log_dir='./logs/tenboard', histogram_freq=0,
#                           write_graph=True, write_images=False)

cbks = [lr_reducer]

model = create_model()
model.summary()

history = model.fit(train_generator, shuffle =True,
                  epochs=eps, workers=4, use_multiprocessing=True,
                  validation_data = val_generator, 
                  verbose = 1,callbacks=cbks)

print('Training complete =========>')
# saving model last weight and the history of the scores
print('Writing the model weights and the history =========>')
model.save(os.path.join(model_path,weight_name))
hist_df = pd.DataFrame(history.history)
hist_csv_file = os.path.join(history_path,hist_name)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
print('Save complete =========>')