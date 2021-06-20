import seaborn as sns
import pandas as pd
import numpy as np
import colorsys
import cv2
from skimage.color import rgb2gray, rgb2hsv


def conc_under_sub_channels(batch): #img is a 3 channels numpy array
    flag = 1
    #temp = batch[0, :, :, :]
    #print(temp.shape)

    for img in batch:
        under_channels = (img == .0).astype(img.dtype)
        over_channels = (img == 1.).astype(img.dtype)

        if img.shape[2] == 3:
            img_conc = np.concatenate((img, under_channels, over_channels), axis=2)
        elif img.shape[0] == 3:
            img_conc = np.concatenate((img, under_channels, over_channels), axis=0)
       
        shape = img_conc.shape
        #print(shape(0))
        
        if flag:
            temp = np.reshape(img_conc, (1, shape[0],shape[1], shape[2]))
            flag = 0
        else:
            img_reshape = np.reshape(img_conc, (1, shape[0],shape[1], shape[2]))
            temp = np.concatenate((temp, img_reshape), axis = 0)
        
        #print(temp.shape)
        #print(result.shape)
        #print('----------')
         
       
    return temp

def hsv_channel(batch):
    flag = 1
    for img in batch:
        if img.shape[2] == 3:
            #hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            #hsv_shape = hsv.shape
            #img_list = img.tolist()
            #img_list.append(hsv[1])
            #img_conc = np.array(img_list)
            #print(img_conc.shape)
            
            hsv = rgb2hsv(img)
            #print(hsv[:,:,1].shape)
            shape = hsv[:,:,1].shape
            #print(np.reshape(hsv[:,:,1], (shape[0],shape[1], 1)).shape)
            #print(shape, img.shape)
            img_conc = np.concatenate((img, np.reshape(hsv[:,:,1], (shape[0],shape[1], 1))), axis = -1)
            
        elif img.shape[0] == 3:
            h, s, v = colorsys.rgb_to_hsv(img[0, :, :], img[1, :, :], img[2, :, :])
            img_conc = np.concatenate((img, s), axis=0)
        shape = img_conc.shape
        #print(shape(0))
        
        if flag:
            temp = np.reshape(img_conc, (1, shape[0],shape[1], shape[2]))
            flag = 0
        else:
            img_reshape = np.reshape(img_conc, (1, shape[0],shape[1], shape[2]))
            temp = np.concatenate((temp, img_reshape), axis = 0)
        
        #print(temp.shape)
        #print(result.shape)
        #print('----------')
         
       
    return temp
        
       
            
        
        

    
