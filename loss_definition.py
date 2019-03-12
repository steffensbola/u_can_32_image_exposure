#Loss definition 

from scipy import stats
import keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops


def loss_mix_v3(y_true, y_pred):
    alpha = 0.2
    l1_w = 1-alpha
    ms_ssim_w = alpha
    
    l1 = K.mean(K.abs(y_pred - y_true)*K.abs(y_true - .5), axis=-1)
    ms_ssim = tf.reduce_mean(1-tf.image.ssim_multiscale(y_pred, y_true, max_val = 1.0))
    
    return ms_ssim_w*ms_ssim + l1_w*l1

