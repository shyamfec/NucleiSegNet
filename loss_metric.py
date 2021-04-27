import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.backend import *

from tensorflow.keras.losses import binary_crossentropy

  ##### Metrices & Loss #####
#------------- Metrice-------------
def f1_score1(y_true,y_pred):
    smooth=1
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
#----------------------------------
# -------------Loss----------------
def dice_loss(y_true, y_pred):
    smooth=1
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    y_true_f = cast(y_true_f, dtype='float32')
    y_pred_f = cast(y_pred_f, dtype='float32')
    intersection = sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

def loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    def dice_coef_loss(y_true, y_pred):
        return dice_loss(y_true, y_pred)
    
    def jaccard_distance_loss(y_true, y_pred, smooth=100):  
        intersection = sum(abs(y_true * y_pred), axis=-1)
        sum_ = sum(abs(y_true) + abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    return (jaccard_distance_loss(y_true, y_pred) * dice_coef_loss(y_true, y_pred))/(jaccard_distance_loss(y_true, y_pred) + dice_coef_loss(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#-----------------------------------
#####################