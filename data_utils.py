

# In[5]:


#################################################################
# Load Adobe MIT FiveK Dataset
#################################################################

import os
import sys
import glob
import numpy as np
import warnings
import shutil
import cv2
# import rawpy
import random
from skimage.io import imsave, imread
from skimage.transform import rescale
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from fnmatch import fnmatch

from multiprocessing import Pool
MAX_N_PROCESS = 6


def center_crop(im, tw, th):
    w, h = im.shape[:2]
    if w == tw and h == th:
        return im

    w_p = int(round((w - tw) / 2))
    h_p = int(round((h - th) / 2))
    return im[w_p:w_p+tw, h_p:h_p+th]

def load_raw(path, scale=1.0):
    # Read RAW Image
    # Returns an image in the 0--1 float range 
    rgb = rescale(rawpy.imread(path).postprocess(), scale, preserve_range=False)
    return rgb

def multiply_and_clip(rgb, threshold = [-.25, .75]):
    # Color
    rgb_clipped = np.clip(rgb, threshold[0], threshold[1])
    rgb_clipped = rgb_clipped *(1+(threshold[0])+(1-threshold[1]))
    rgb_clipped = np.clip(rgb_clipped, .0, 1.)
    return rgb_clipped

def saturate(rgb, threshold = [0, .6]):
    rgb_clipped = np.clip(rgb, threshold[0], threshold[1])
    rgb_clipped = rgb_clipped- np.min(rgb_clipped)
    rgb_clipped = rgb_clipped/np.max(rgb_clipped)
    rgb_clipped = np.clip(rgb_clipped, .0, 1.)
    return rgb_clipped

    
def add_noise(image, noise_typ = "gauss", mean = 0, var = 0.1, exp_sigma = 0.5, percent = 1):
    
    if noise_typ == "gauss":
        row,col,ch= image.shape
        #mean = 0
        #var = 0.1
        sigma = ((percent**2)*var)**exp_sigma
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy = np.clip(noisy, .0, 1.)
        return noisy
    
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        out = np.clip(out, .0, 1.)
        return out
    
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        noisy = np.clip(noisy, .0, 1.)
        return noisy
    
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        noisy = np.clip(noisy, .0, 1.)
        return noisy
    elif noise_typ == "glare":
        print('Not yet implemented')
        
class DataLoaderDistort():
    def __init__(self, datases_path, dataset_name, img_res=(256, 256), train_split = .7,
                 distortion=['low_range','saturate', 'underexpose', 'underexpose_p25','gauss', 'speckle', 'poisson', 's&p'], 
                 filetype = "*.dng", percent_noise_gt = 0.0,
                 debug=False):
        np.random.RandomState(seed=42)
        self.raw_types = ["*.dng","*.DNG","*.arw",".ARW"]
        self.filetype = filetype
        self.dataset_name = dataset_name
        self.distortion = distortion
        self.img_res = img_res
        self.datases_path = datases_path
        self.debug = debug
        self.path =  os.path.join(datases_path, dataset_name)
        self.imgs_path = []
        self.percent_noise_gt = percent_noise_gt
        for paths, subdirs, files in os.walk(self.path):
            for name in files:
                if fnmatch(name, self.filetype):
                    self.imgs_path.append(os.path.join(paths, name))
                    
        self.train_imgs_path = self.imgs_path[:int(np.floor(train_split*len(self.imgs_path)))]
        self.test_imgs_path = self.imgs_path[int(np.floor((1.0-train_split)*len(self.imgs_path))):]

        final_train_path = os.path.join(self.path, 'tif', 'train')
        final_test_path = os.path.join(self.path, 'tif', 'test')

        if not os.path.isdir(final_train_path):
            os.makedirs(final_train_path)
            for train_img in self.train_imgs_path:
                img = train_img.split("/")[-1]
                #print(os.path.join(final_train_path, img))
                os.rename(train_img, os.path.join(final_train_path, img))
        
        
        if not os.path.isdir(final_test_path):
            self.test_imgs_path = glob.glob(os.path.join(self.path, 'tif', '*.tif') )
            os.makedirs(final_test_path)
            for test_img in self.test_imgs_path:
                img = test_img.split("/")[-1]
                #print(os.path.join(final_test_path, img))
                os.rename(test_img, os.path.join(final_test_path, img))
    #def separate_data(self):




        
    def load_data(self, batch_size=1, is_testing=False):
        imgs_A = []
        imgs_B = []
        
        if is_testing:
            rnd = np.random.randint(0, len(self.test_imgs_path)- batch_size) 
            rnd = range(rnd, rnd+int(np.ceil(batch_size/len(self.distortion))))
        else:
            rnd = np.random.randint(0, len(self.train_imgs_path)- batch_size)
            rnd = range(rnd, rnd+int(np.ceil(batch_size/len(self.distortion))))
        for j in rnd:
            flag_pass = 1
            if is_testing:
                try:
                    if self.filetype in self.raw_types:
                        img_A = resize(load_raw(self.test_imgs_path[j]), self.img_res)
                    else:
                        img_A = resize(imread(self.test_imgs_path[j])/255., self.img_res)
                except:
                    print("Deu ruim por causa imagem corrompida nos testes!!!!")
                    print(self.train_imgs_path[j])
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                
            if not is_testing:
                try:
                    if self.filetype in self.raw_types:
                        img_A = resize(load_raw(self.train_imgs_path[j]), self.img_res)
                    else:
                        img_A = resize(imread(self.train_imgs_path[j])/255., self.img_res)   
                except:
                    print("Deu ruim por causa imagem corrompida no treinamento!!!!")
                    print(self.train_imgs_path[j])
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                # print(self.train_imgs_path[j])
                #img_A = resize(load_raw(self.train_imgs_path[j]), self.img_res)
            
                
                # print(img_A)
            #noise addition  
           
            if flag_pass:
                 #Choose between ground truth with noise or not. Always gausian noise. 
                if self.percent_noise_gt > 0.:
                    imgs_A.append(add_noise(img_A, 'gauss', percent = self.percent_noise_gt))
                else:
                    imgs_A.append(img_A)
                  
                if 'low_range' in self.distortion:
                    imgs_B.append(multiply_and_clip(img_A))
                if 'saturate' in self.distortion:
                    imgs_B.append(saturate(img_A,[.0,.6]))
                if 'underexpose' in self.distortion:
                    imgs_B.append(saturate(img_A,[.4,1.]))
                if 'underexpose_p5' in self.distortion:
                    imgs_B.append(saturate(img_A,[np.percentile(img_A,5),1.]))
                if 'underexpose_p10' in self.distortion:
                    imgs_B.append(saturate(img_A,[np.percentile(img_A,10),1.]))
                if 'underexpose_p15' in self.distortion:
                    imgs_B.append(saturate(img_A,[np.percentile(img_A,15),1.]))
                if 'underexpose_p25' in self.distortion:
                    imgs_B.append(saturate(img_A,[np.percentile(img_A,25),1.]))
                if 'saturate_p5' in self.distortion:
                    imgs_B.append(saturate(img_A,[0.,np.percentile(img_A,95)]))
                if 'saturate_p10' in self.distortion:
                    imgs_B.append(saturate(img_A,[0.,np.percentile(img_A,90)]))
                if 'saturate_p15' in self.distortion:
                    imgs_B.append(saturate(img_A,[0.,np.percentile(img_A,85)]))
                if 'saturate_p25' in self.distortion:
                    imgs_B.append(saturate(img_A,[0.,np.percentile(img_A,75)]))
                if 'denoise_tv_chambolle' in self.distortion:
                    imgs_B.append(denoise_tv_chambolle(img_A, weight=10))
                if 'denoise_bilateral' in self.distortion:
                    imgs_B.append(denoise_bilateral(img_A, weight=10))    
                if 'gauss' in self.distortion:
                    imgs_B.append(add_noise(img_A, 'gauss'))
                if 'speckle' in self.distortion:
                    imgs_B.append(add_noise(img_A, 'speckle'))
                if 'poisson' in self.distortion:
                    imgs_B.append(add_noise(img_A, 'poisson'))
                if 's&p' in self.distortion:
                    imgs_B.append(add_noise(img_A, 's&p'))  
#            if not flag_pass:
#                j = j-1 #disconsider the actual interaction
            
        imgs_A = np.array(imgs_A[:batch_size])
        imgs_B = np.array(imgs_B[:batch_size])
        
        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_B, imgs_A

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        pathA = '%s/%s/%s/A/*' % (self.datases_path, self.dataset_name, data_type)
        if self.debug:
            print(pathA)
        pathA = glob(pathA)
        pathB = '%s/%s/%s/B/*' % (self.datases_path, self.dataset_name, data_type)
        if self.debug:
            print(pathB)
        pathB = glob(pathB)
        
        if self.debug:
            print(pathA)
            print(pathB)

        self.n_batches = int(len(pathA) / batch_size)

        for i in range(self.n_batches-1):
            batchIdx = range(i*batch_size,(i+1)*batch_size,1)

            imgs_A, imgs_B = [], []
            for j in batchIdx:
                img_A = self.imread(pathA[j])
                img_B = self.imread(pathB[j])

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)
                
                #print(img_A.shape, img_B.shape)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

# a = DataLoaderFiveK(DATASET_FOLDER, DATASET_NAME)
# y,x = a.load_data()

# for i in range(0,len(x)):
#     imsave(os.path.join(LOG_FOLDER, str(i).zfill(6)+'x.png'), x[i])
#     imsave(os.path.join(LOG_FOLDER, str(i).zfill(6)+'y.png'), y[i])


################################################################################
# Data Loader
################################################################################
class DataLoader():
    def __init__(self, datases_path, dataset_name, img_res=(256, 256), debug=False):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datases_path = datases_path
        self.debug = debug


    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        pathA = '%s/%s/%s/A/*' % (self.datases_path, self.dataset_name, data_type)
        if self.debug:
            print(pathA)
        pathA = glob(pathA)
        if self.debug:
            print(pathA)
        pathB = '%s/%s/%s/B/*' % (self.datases_path, self.dataset_name, data_type)
        if self.debug:
            print(pathB)
        pathB = glob(pathB)

        imgs_A = []
        imgs_B = []
        rnd = np.random.randint(0, len(pathA), batch_size)
        for j in rnd:
            img_A = self.imread(pathA[j])
            img_B = self.imread(pathB[j])

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        
        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        pathA = '%s/%s/%s/A/*' % (self.datases_path, self.dataset_name, data_type)
        if self.debug:
            print(pathA)
        pathA = glob(pathA)
        pathB = '%s/%s/%s/B/*' % (self.datases_path, self.dataset_name, data_type)
        if self.debug:
            print(pathB)
        pathB = glob(pathB)
        
        if self.debug:
            print(pathA)
            print(pathB)

        self.n_batches = int(len(pathA) / batch_size)

        for i in range(self.n_batches-1):
            batchIdx = range(i*batch_size,(i+1)*batch_size,1)

            imgs_A, imgs_B = [], []
            for j in batchIdx:
                img_A = self.imread(pathA[j])
                img_B = self.imread(pathB[j])

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    

################################################################################
# Data Loader for Learning to See in the Dark Dataset
################################################################################
class DataLoaderSeeInTheDark():
    def __init__(self, dataset_path, dataset_name, img_res=(256, 256), debug=False):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.dataset_path = dataset_path
        self.debug = debug
        self.path =  os.path.join(dataset_path, dataset_name)
        with open(os.path.join(self.path,dataset_name+'_train_list.txt')) as f:
            self.train_imgs_path = [line.split() for line in f]
        with open(os.path.join(self.path,dataset_name+'_test_list.txt')) as f:
            self.test_imgs_path = [line.split() for line in f]
        


    def load_data(self, batch_size=1, is_testing=False):
        imgs_A = []
        imgs_B = []
        img_B = None
        img_A = None
        
        if is_testing:
            rnd = np.random.random_integers(0, len(self.test_imgs_path)-1, batch_size) 
            
        else:
            rnd = np.random.random_integers(0, len(self.train_imgs_path)-1, batch_size)
            
        for j in rnd:
            flag_pass = 1
            if is_testing:
                try:
                  img_A = load_raw(os.path.join(self.path, self.test_imgs_path[j][0]))                  
                  img_B = load_raw(os.path.join(self.path, self.test_imgs_path[j][1]))

                except:
                    print("Deu ruim por causa imagem corrompida nos testes!!!!")
                    print(os.path.join(self.path, self.test_imgs_path[j][0]))
                    print(os.path.join(self.path, self.test_imgs_path[j][1]))
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                
            if not is_testing:
                try:
                  img_A = load_raw(os.path.join(self.path, self.train_imgs_path[j][0]))                  
                  img_B = load_raw(os.path.join(self.path, self.train_imgs_path[j][1]))
                 
                except:
                    print("Deu ruim por causa imagem corrompida no treinamento!!!!")
                    print(os.path.join(self.path, self.train_imgs_path[j][0]))
                    print(os.path.join(self.path, self.train_imgs_path[j][1]))
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
            
            if flag_pass == 1:
                if not self.img_res is None:
                      img_A = resize(img_A, self.img_res)
                      img_B = resize(img_B, self.img_res)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
                  
                  
                  
        return np.array(imgs_A), np.array(imgs_B)

    
class DataLoaderSteffens2018():
    def __init__(self, datases_path, dataset_name='all_small', img_res=(256, 256),  train_split = .7, filetype = "*.JPG", a_folder='u', b_folder='hdr', debug=False, get_all = False):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datases_path = os.path.join(datases_path, dataset_name)
        self.debug = debug
        self.filetype = filetype
        print(self.datases_path)
        pathA = []
        pathB = []
        for paths, subdirs, files in os.walk(os.path.join(self.datases_path,a_folder)):
            for name in files:
                if fnmatch(name, self.filetype):
                    pathA.append(os.path.join(paths, name))
        
        for paths, subdirs, files in os.walk(os.path.join(self.datases_path,b_folder)):
            for name in files:
                if fnmatch(name, self.filetype):
                    pathB.append(os.path.join(paths, name))
                    
        pathA = sorted(pathA)
        pathB = sorted(pathB)
        
        if get_all:
            train_split = 1
        
        print('train_split:',train_split)
        
        
        self.imgs_path = list(zip(pathA, pathB))
        
        self.train_imgs_path = self.imgs_path[:int(np.floor(train_split*len(self.imgs_path)))]
        self.test_imgs_path = self.imgs_path[len(self.imgs_path)-int(np.ceil((1.0-train_split)*len(self.imgs_path))):]
        
        if self.debug:
            print(self.train_imgs_path)
            print(self.test_imgs_path)
        assert(len(pathA) == len(pathB))
        

    def load_data(self, batch_size=1, is_testing=False, get_all = False):
        #self.test_imgs_path = 
            

        imgs_A = []
        imgs_B = []
        
        if get_all:
            is_testing = False 
            #print(is_testing)
        
        if not get_all:
            if is_testing:
                rnd = np.random.randint(0, len(self.test_imgs_path)- (1+batch_size)) 
                rnd = range(rnd, rnd+int(np.ceil(batch_size)))
            else:
                rnd = np.random.randint(0, len(self.train_imgs_path)- (1+batch_size)) 
                rnd = range(rnd, rnd+int(np.ceil(batch_size)))
        else:
            #rnd = np.random.randint(0, batch_size-1)
            #rnd = range(rnd, rnd+int(np.ceil(batch_size)))
            rnd = range(batch_size)
            print(rnd)
        for j in rnd:
            flag_pass = 1
            if is_testing:
                #print('1')
                try:
                    img_A = imread(self.test_imgs_path[j][0])/255. if self.img_res==None else resize(imread(self.test_imgs_path[j][0])/255., self.img_res)
                    img_B = imread(self.test_imgs_path[j][1])/255. if self.img_res==None else resize(imread(self.test_imgs_path[j][1])/255., self.img_res)
                except:
                    print(sys.exc_info()[0])
                    print("Deu ruim por causa imagem corrompida nos testes!!!!")
                    print(self.train_imgs_path[j])
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                
            if not is_testing:
                #print('2')
                try:
                    img_A = imread(self.train_imgs_path[j][0])/255. if self.img_res==None else resize(imread(self.train_imgs_path[j][0])/255., self.img_res)
                    img_B = imread(self.train_imgs_path[j][1])/255. if self.img_res==None else resize(imread(self.train_imgs_path[j][1])/255., self.img_res)  
                except:
                    print(sys.exc_info())
                    print("Deu ruim por causa imagem corrompida no treinamento!!!!")
                    print(self.train_imgs_path[j])
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
            if flag_pass:
                imgs_A.append(img_A)
                imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        
        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

class DataLoaderCai2018():
    def __init__(self, datases_path, dataset_name='Cai2018_Part1', img_res=None, dark_bright = .1, train_split = .8, debug=False, sortway = 1):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datases_path = os.path.join(datases_path, dataset_name)
        self.debug = debug
        self.filetype = ".JPG"
        self.dark_bright = dark_bright
        print(self.datases_path)
        self.pathA = os.path.join(self.datases_path,'input')
        self.pathB = os.path.join(self.datases_path,'Label')
        
        self.imgs_path = os.listdir(self.pathA)
        
        if sortway == 1:
                
            self.test_imgs_path = self.imgs_path[int(np.floor((1.0-train_split)*len(self.imgs_path))):]
        elif sortway == 2:
            self.train_imgs_path = []
            self.test_imgs_path = []
            files_n = len(self.imgs_path)
            train_n = train_split*files_n   
            stride = files_n//train_n
            j = 0
            #print(self.imgs_path)
            #print(train_n)
            #print(files_n)
            #print(stride)
            flag = 0
            for i in range(len(self.imgs_path)):
                #print(self.test_imgs_path)
                #print(str(self.imgs_path[i]))
                if i == j*stride and flag == 0:
                    #print(i)
                    j = j+1    
                    self.train_imgs_path.append(str(self.imgs_path[i]))
                    #os.rename(dataset+files[i], dataset+'/test/'+files[i])
                    if j >= train_n:
                        flag = 1
                else:
                    self.test_imgs_path.append(str(self.imgs_path[i]))
        
        if self.debug:
            print('train:')
            print(self.train_imgs_path)
            print('test:')
            print(self.test_imgs_path)
        
        input_path = os.path.join(datases_path, self.dataset_name, "input")
        label_path = os.path.join(datases_path, self.dataset_name, "Label")
        
        paths = [input_path, label_path]
        flag = 1
        for path in paths:
            final_train_path = os.path.join(path, "train")
            final_test_path = os.path.join(path, "test")
            #print(final_train_path)

            if not os.path.isdir(final_train_path):
                os.makedirs(final_train_path)
                for train_img in self.train_imgs_path:
                    if flag:
                        shutil.move(os.path.join(path, train_img), final_train_path)
                    else:
                        shutil.move(os.path.join(path, train_img)+self.filetype, final_train_path)
            
            #print(self.test_imgs_path)
            if not os.path.isdir(final_test_path):
                os.makedirs(final_test_path)
                for test_img in self.test_imgs_path:
                    if flag:
                        shutil.move(os.path.join(path, test_img), final_test_path)
                    else:
                        shutil.move(os.path.join(path, test_img)+self.filetype, final_test_path)
            flag = 0

    def load_data2(self, batch_size=1, is_testing=False):
        imgs_A = []
        imgs_B = []

        if is_testing:
            test_path = os.listdir(os.path.join(self.pathA, 'test'))
            #print(test_path)
            rnd = np.random.randint(0, len(test_path)- (1+batch_size)) 
            rnd = range(rnd, rnd+int(np.ceil(batch_size)))
            #print(rnd)
        else:
            train_path = os.listdir(os.path.join(self.pathA, 'train'))
            #print(train_path)
            rnd = np.random.randint(0, len(train_path)- (1+batch_size)) 
            rnd = range(rnd, rnd+int(np.ceil(batch_size)))
            #print(rnd)
        #print(rnd)
        for j in rnd:
            flag_pass = 1
            if is_testing:
                #print(test_path[j])
                #print(os.path.join(os.path.join(self.pathA, 'test'), test_path[j]))
                files_in_folder = len(os.listdir(os.path.join(os.path.join(self.pathA, 'test'), test_path[j])))
                file_path = [os.path.join(os.path.join(self.pathA, 'test'), test_path[j], str(1+int(files_in_folder*self.dark_bright)).zfill(3)+self.filetype), os.path.join(self.pathB, 'test',test_path[j]+self.filetype)]
                try:
                    img_A = imread(file_path[0])/255. if self.img_res==None else resize(imread(file_path[0])/255., self.img_res)
                    img_B = imread(file_path[1])/255. if self.img_res==None else resize(imread(file_path[1])/255., self.img_res)
                except:
                    print(sys.exc_info()); print("Deu ruim por causa imagem corrompida nos testes!!!!"); print(file_path)
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                #print(img_A)
                #print(img_B
            else:
                #print(train_path[j])
                #print(os.path.join(os.path.join(self.pathA, 'train'), train_path[j]))
                files_in_folder = len(os.listdir(os.path.join(os.path.join(self.pathA, 'train'), train_path[j])))
                file_path = [os.path.join(os.path.join(self.pathA, 'train'), train_path[j], str(1+int(files_in_folder*self.dark_bright)).zfill(3)+self.filetype), os.path.join(self.pathB, 'train',train_path[j]+self.filetype)]
                try:
                    img_A = imread(file_path[0])/255. if self.img_res==None else resize(imread(file_path[0])/255., self.img_res)
                    img_B = imread(file_path[1])/255. if self.img_res==None else resize(imread(file_path[1])/255., self.img_res)
                except:
                    print(sys.exc_info()); print("Deu ruim por causa imagem corrompida no traino!!!!"); print(file_path)
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                #print(img_A)
                #print(img_B)
            if flag_pass:
                imgs_A.append(img_A)
                imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        
        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_A, imgs_B

    def load_data(self, batch_size=1, is_testing=False):
        

        imgs_A = []
        imgs_B = []
        
        if is_testing:
            rnd = np.random.randint(0, len(self.test_imgs_path)- (1+batch_size)) 
            rnd = range(rnd, rnd+int(np.ceil(batch_size)))
        else:
            rnd = np.random.randint(0, len(self.train_imgs_path)- (1+batch_size)) 
            rnd = range(rnd, rnd+int(np.ceil(batch_size)))
        for j in rnd:
            flag_pass = 1
            if is_testing:
                files_in_folder = len(os.listdir(os.path.join(self.pathA, self.test_imgs_path[j])))
                file_path = [os.path.join(self.pathA, self.test_imgs_path[j], str(1+int(files_in_folder*self.dark_bright)).zfill(3)+self.filetype), os.path.join(self.pathB, self.test_imgs_path[j]+self.filetype)]
                try:
                    img_A = imread(file_path[0])/255. if self.img_res==None else resize(imread(file_path[0])/255., self.img_res)
                    img_B = imread(file_path[1])/255. if self.img_res==None else resize(imread(file_path[1])/255., self.img_res)
                except:
                    print(sys.exc_info()); print("Deu ruim por causa imagem corrompida nos testes!!!!"); print(file_path)
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
                
            if not is_testing:
                files_in_folder = len(os.listdir(os.path.join(self.pathA, self.train_imgs_path[j])))
                file_path = [os.path.join(self.pathA, self.train_imgs_path[j],  str(1+int(files_in_folder*self.dark_bright)).zfill(3)+self.filetype), os.path.join(self.pathB, self.train_imgs_path[j]+self.filetype)]
                try:
                    img_A = imread(file_path[0])/255. if self.img_res==None else resize(imread(file_path[0])/255., self.img_res)
                    img_B = imread(file_path[1])/255. if self.img_res==None else resize(imread(file_path[1])/255., self.img_res)  
                except:
                    print(sys.exc_info()); print("Deu ruim por causa imagem corrompida no treinamento!!!!"); print(file_path)
                    flag_pass = 0 #if didn't load the image, don't pass through noise addition.
                    pass
            if flag_pass:
                imgs_A.append(img_A)
                imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        
        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_A, imgs_B
    
    


    def imread(self, path):
        skimage.io.imread(path, plugin='matplotlib')
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    # invGamma = 1.0 / gamma
    # table = np.array([((i / 255.0) ** invGamma) * 255
    #                   for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table

    return (image)**(1/gamma)

class dataloader():
    def __init__(self, dataset_name: object, saturation: object, dataset_path: object = None, img_res: object = None, filetype: object = None,
                 percent_saturation: object = 0,
                 gamma = 1.0,
                 percent_noise: object = 0,
                 debug: object = False) -> object:
        # TODO: recive a saturation list
        self.datasets = {"Cai2018_Part1" : "/mnt/dados/datasets/Cai2018_Part1",
            "fivek" : "/mnt/dados/datasets/fivek_dataset/tif",
            "hdr+burst" : "/mnt/dados/datasets/hdr+burst",
            "a6300" : "/mnt/dados/datasets/a6300_exposure/all_small"}
        self.raw_types = ["*.dng", "*.DNG", "*.arw", ".ARW"]
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.filetype = filetype
        self.percent_saturation = percent_saturation
        self.gamma = gamma
        self.saturation = saturation
        self.percent_noise_gt = percent_noise
        self.debug = debug

        if self.filetype == None:
            if self.dataset_name == "Cai2018_Part1":
                self.filetype = ".JPG"
            elif self.dataset_name == "fivek":
                self.filetype = ".tif"

            elif self.dataset_name == "hdr+burst":
                self.filetype = ".jpg"

            elif self.dataset_name == "a6300":
                self.filetype = ".JPG"

        if (self.dataset_name == "fivek" or self.dataset_name == "hdr+burst"):
            if not isinstance(self.saturation, (list,)):
                self.saturation = [self.saturation]
            if not isinstance(self.gamma, (list,)):
                self.gamma = [self.gamma]


        if not dataset_path == None:
            self.datasets[self.dataset_name] = os.path.join(dataset_path, dataset_name)

        if self.debug:
            print("Dataset name:", self.dataset_name)
            print("Dataset path:", self.datasets.get(self.dataset_name))

    def load_data (self, batch_size = 1, is_testing = False, path_pass = False):
        if self.dataset_name == "Cai2018_Part1":
            return self.load_cai(batch_size = batch_size, is_testing = is_testing, path_pass = path_pass)
        elif self.dataset_name == "a6300":
            return self.load_a6300(batch_size = batch_size, is_testing = is_testing, path_pass = path_pass)
        else:
            return self.load_default(batch_size = batch_size, is_testing = is_testing, path_pass = path_pass)

    def load_default(self, batch_size, is_testing=False, path_pass = False):
        imgs_A = []
        imgs_B = []
        print("batch_size:",batch_size)
        if is_testing:
            test_imgs_path = glob.glob(os.path.join(self.datasets[self.dataset_name], 'test', '*'+self.filetype))
            #print(os.path.join(self.datasets[self.dataset_name], 'test', '*'+self.filetype))
            #print(test_imgs_path)
            if (len(test_imgs_path) - batch_size > 0):
                rnd = np.random.randint(0, len(test_imgs_path) - batch_size)
                rnd = range(rnd, rnd + int(np.ceil(batch_size / len(self.saturation))))
            else:
                rnd = range(0, len(test_imgs_path))
                #rnd = range(len(test_imgs_path)-1, len(test_imgs_path))
        else:
            train_imgs_path = glob.glob(os.path.join(self.datasets[self.dataset_name], 'train', '*'+self.filetype))
            rnd = np.random.randint(0, len(train_imgs_path) - batch_size)
            rnd = range(rnd, rnd + int(np.ceil(batch_size / len(self.saturation))))
        print("rnd:", rnd)
        for j in rnd:
            # print('J:', j)
            # print(len(self.saturation))
            flag_pass = 1
            if is_testing:
                try:
                    if not path_pass:
                        if self.filetype in self.raw_types:
                            #img_A = imread(os.path.join(pathA, 'test',test_path[j])) / 255. if self.img_res == None else resize(imread(os.path.join(pathA,'test',test_path[j])) / 255., self.img_res)
                            img_A = resize(load_raw(test_imgs_path[j]), self.img_res) if self.img_res != None else load_raw(test_imgs_path[j])
                        else:
                            img_A = resize(imread(test_imgs_path[j]) / 255., self.img_res) if self.img_res != None else imread(test_imgs_path[j]) / 255.
                    else:
                        img_A = test_imgs_path[j]
                except:
                    print("Deu ruim por causa imagem corrompida nos testes!!!!")
                    print(test_imgs_path[j])
                    flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                    pass

            if not is_testing:
                try:
                    if not path_pass:
                        if self.filetype in self.raw_types:
                            img_A = resize(load_raw(train_imgs_path[j]), self.img_res) if self.img_res != None else load_raw(train_imgs_path[j])
                        else:
                            img_A = resize(imread(train_imgs_path[j]) / 255.,self.img_res) if self.img_res != None else imread(train_imgs_path[j]) / 255.
                    else:
                        img_A = test_imgs_path[j]
                except:
                    print("Deu ruim por causa imagem corrompida no treinamento!!!!")
                    print(train_imgs_path[j])
                    flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                    pass
                # print(self.train_imgs_path[j])
                # img_A = resize(load_raw(self.train_imgs_path[j]), self.img_res)

                # print(img_A)
            if not path_pass:
                # noise addition

                img_A_gt = img_A
                #img_A = adjust_gamma(img_A, gamma=self.gamma)
                gamma = random.choice(self.gamma)
                if self.debug:
                    print("LIST: ", self.gamma)
                    print("GaMMA: ", gamma)
                img_A = (img_A)**(1/gamma)

                if flag_pass:
                    # Choose between ground truth with noise or not. Always gausian noise.
                    if self.percent_noise_gt > 0.:
                        imgs_A.append(add_noise(img_A_gt, 'gauss', percent=self.percent_noise_gt))
                    else:
                        imgs_A.append(img_A_gt)

                    if 'low_range' in self.saturation:
                        imgs_B.append(multiply_and_clip(img_A))
                    # if 'saturate' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [.0, .6]))
                    # if 'underexpose' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [.4, 1.]))
                    # if 'underexpose_p5' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [np.percentile(img_A, 5), 1.]))
                    # if 'underexpose_p10' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [np.percentile(img_A, 10), 1.]))
                    # if 'underexpose_p15' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [np.percentile(img_A, 15), 1.]))
                    # if 'underexpose_p25' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [np.percentile(img_A, 25), 1.]))
                    # if 'saturate_p5' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [0., np.percentile(img_A, 95)]))
                    # if 'saturate_p10' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [0., np.percentile(img_A, 90)]))
                    # if 'saturate_p15' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [0., np.percentile(img_A, 85)]))
                    # if 'saturate_p25' in self.saturation:
                    #     imgs_B.append(saturate(img_A, [0., np.percentile(img_A, 75)]))
                    if 'over' in self.saturation:
                        imgs_B.append(saturate(img_A, [0., np.percentile(img_A, 100 - self.percent_saturation)]))
                    if 'under' in self.saturation:
                        imgs_B.append(saturate(img_A, [np.percentile(img_A, self.percent_saturation), 1.]))
                    if 'denoise_tv_chambolle' in self.saturation:
                        imgs_B.append(denoise_tv_chambolle(img_A, weight=10))
                    if 'denoise_bilateral' in self.saturation:
                        imgs_B.append(denoise_bilateral(img_A, weight=10))
                    if 'gauss' in self.saturation:
                        imgs_B.append(add_noise(img_A, 'gauss'))
                    if 'speckle' in self.saturation:
                        imgs_B.append(add_noise(img_A, 'speckle'))
                    if 'poisson' in self.saturation:
                        imgs_B.append(add_noise(img_A, 'poisson'))
                    if 's&p' in self.saturation:
                        imgs_B.append(add_noise(img_A, 's&p'))
                        #            if not flag_pass:
            #                j = j-1 #disconsider the actual interaction
            else:
                imgs_A.append(img_A)
                imgs_B.append(img_A)

        imgs_A = np.array(imgs_A[:batch_size])
        imgs_B = np.array(imgs_B[:batch_size])

        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_B, imgs_A

    def load_a6300(self, batch_size=1, is_testing=False, path_pass = False):
        imgs_A = []
        imgs_B = []

        if self.saturation == 'over':
            folder_name = 'o'

        elif self.saturation == 'under':
            folder_name = 'u'

        pathA = os.path.join(self.datasets[self.dataset_name], folder_name)
        pathB = os.path.join(self.datasets[self.dataset_name], 'c')

        if is_testing:
            test_path = os.listdir(os.path.join(pathA, 'test'))
            if (len(test_path) - (1 + batch_size) > 0):
                #rnd = np.random.randint(0, len(test_path) - (1 + batch_size))
                rnd = np.random.randint(0, len(test_path) - batch_size)
                rnd = range(rnd, rnd + int(np.ceil(batch_size)))
            else:
                rnd = range(0, len(test_path))
                #rnd = range(len(test_path)-1, len(test_path))

        else:
            train_path = os.listdir(os.path.join(pathA, 'train'))
            rnd = np.random.randint(0, len(train_path) - (1 + batch_size))
            rnd = range(rnd, rnd + int(np.ceil(batch_size)))
        print(rnd)
        for j in rnd:
            flag_pass = 1
            if is_testing:
                # print('1')
                try:
                    if not path_pass:
                        img_A = imread(os.path.join(pathA, 'test',test_path[j])) / 255. if self.img_res == None else resize(
                            imread(os.path.join(pathA,'test',test_path[j])) / 255., self.img_res)
                        img_B = imread(os.path.join(pathB,'test',test_path[j])) / 255. if self.img_res == None else resize(
                            imread(os.path.join(pathB,'test',test_path[j])) / 255., self.img_res)
                    else:
                        img_A = os.path.join(pathA, 'test',test_path[j])
                        img_B = os.path.join(pathB,'test',test_path[j])
                except:
                    print(sys.exc_info()[0])
                    print("Deu ruim por causa imagem corrompida nos testes!!!!")
                    print(test_path[j])
                    flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                    pass

            if not is_testing:
                # print('2')
                try:
                    if not path_pass:
                        img_A = imread(os.path.join(pathA, 'train',train_path[j])) / 255. if self.img_res == None else resize(
                            imread(os.path.join(pathA, 'train',train_path[j])) / 255., self.img_res)
                        img_B = imread(os.path.join(pathB, 'train',train_path[j])) / 255. if self.img_res == None else resize(
                            imread(os.path.join(pathB, 'train',train_path[j])) / 255., self.img_res)
                    else:
                        img_A = os.path.join(pathA, 'train', test_path[j])
                        img_B = os.path.join(pathB, 'train', test_path[j])
                except:
                    print(sys.exc_info())
                    print("Deu ruim por causa imagem corrompida no treinamento!!!!")
                    print(train_path[j])
                    flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                    pass
            if flag_pass:
                imgs_A.append(img_A)
                imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)

        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_A, imgs_B

    def load_cai(self, batch_size=1, is_testing=False, path_pass = False, dark_bright = None):
        imgs_A = []
        imgs_B = []
        pathA = os.path.join(self.datasets[self.dataset_name], 'input')
        pathB = os.path.join(self.datasets[self.dataset_name], 'Label')

        if (self.saturation == 'over' and dark_bright == None):
            dark_bright = .9

        elif (self.saturation == 'under' and dark_bright == None):
            dark_bright = .1

        if is_testing:
            test_path = os.listdir(os.path.join(pathA, 'test'))
            if (len(test_path) - (1 + batch_size) > 0):
                #rnd = np.random.randint(0, len(test_path) - (1 + batch_size))
                rnd = np.random.randint(0, len(test_path) - batch_size)
                rnd = range(rnd, rnd + int(np.ceil(batch_size)))
            else:
                #rnd = range(len(test_path)-1, len(test_path))
                rnd = range(0, len(test_path))
        else:
            train_path = os.listdir(os.path.join(pathA, 'train'))
            rnd = np.random.randint(0, len(train_path) - (1 + batch_size))
            rnd = range(rnd, rnd + int(np.ceil(batch_size)))
        print(rnd)
        for j in rnd:
            #print('J:', j)
            flag_pass = 1
            if is_testing:
                files_in_folder = len(os.listdir(os.path.join(os.path.join(pathA, 'test'), test_path[j])))
                file_path = [os.path.join(os.path.join(pathA, 'test'), test_path[j],
                                          str(1 + int(files_in_folder * dark_bright)).zfill(
                                              3) + self.filetype),
                             os.path.join(pathB, 'test', test_path[j] + self.filetype)]
                try:
                    if not path_pass:
                        img_A = imread(file_path[0]) / 255. if self.img_res == None else resize(
                            imread(file_path[0]) / 255., self.img_res)
                        img_B = imread(file_path[1]) / 255. if self.img_res == None else resize(
                            imread(file_path[1]) / 255., self.img_res)
                    else:
                        img_A = file_path[0]
                        img_B = file_path[1]
                except:
                    print(sys.exc_info());
                    print("Deu ruim por causa imagem corrompida nos testes!!!!");
                    print(file_path)
                    flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                    pass
            else:
                files_in_folder = len(os.listdir(os.path.join(os.path.join(pathA, 'train'), train_path[j])))
                file_path = [os.path.join(os.path.join(pathA, 'train'), train_path[j],
                                          str(1 + int(files_in_folder * dark_bright)).zfill(
                                              3) + self.filetype),
                             os.path.join(pathB, 'train', train_path[j] + self.filetype)]
                try:
                    if not path_pass:
                        img_A = imread(file_path[0]) / 255. if self.img_res == None else resize(
                            imread(file_path[0]) / 255., self.img_res)
                        img_B = imread(file_path[1]) / 255. if self.img_res == None else resize(
                            imread(file_path[1]) / 255., self.img_res)
                    else:
                        img_A = file_path[0]
                        img_B = file_path[1]
                except:
                    print(sys.exc_info());
                    print("Deu ruim por causa imagem corrompida no treino!!!!");
                    print(file_path)
                    flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                    pass
            if flag_pass:
                imgs_A.append(img_A)
                imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)

        if self.debug:
            print(imgs_A.shape)
            print(imgs_B.shape)

        return imgs_A, imgs_B


class blind_dataloader():
    def __init__(self, dataset_path=None, img_res: object = None, filetype=".JPG", debug=False) -> object:
        self.dataset_path = dataset_path
        self.img_res = img_res
        self.filetype = filetype
        self.debug = debug

    def load(self, batch_size = 1):
        imgs = []
        img_list = glob.glob(os.path.join(self.dataset_path, '*'+self.filetype))

        for j in range(batch_size):
            flag_pass = 1
            try:
                img = imread(img_list[j]) / 255. if self.img_res == None else resize(imread(img_list[j]) / 255., self.img_res)
            except:
                print(sys.exc_info());
                print("Deu ruim !!!!");
                print(img_list)
                flag_pass = 0  # if didn't load the image, don't pass through noise addition.
                pass
            if flag_pass:
                imgs.append(img)
        imgs = np.array(imgs)

        if self.debug:
            print(imgs.shape)

        return imgs




