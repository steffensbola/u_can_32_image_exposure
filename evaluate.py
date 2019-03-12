import argparse
from skimage.io import imsave, imread	
import keras
from keras.models import load_model
import numpy as np 
import loss_definition
from skimage.transform import resize
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "Input image")
parser.add_argument("--output", "-o", help = "Output image")
parser.add_argument("--default", "-d", help = "Load a default image", action = "store_true")
parser.add_argument("degradation", help = "Seletc the problem type: under or over", choices=["under", "over"])
parser.add_argument("--rows", "-r", help = "Input height")
parser.add_argument("--cols", "-c", help = "Input width")

args = parser.parse_args()

#Read a defaut input image or a passed input image and setup the network model and weigths
if args.degradation == 'under':
	img = imread('under_test.JPG')
	model_path = 'model_under.hd5f'
	weights_path = 'weigths_under.hd5f'

elif args.degradation == 'over':
	img = imread('over_test.JPG')
	model_path = 'model_over.hd5f'
	weights_path = 'weigths_over.hd5f'

if not args.default:
	img = imread(args.input)

if img.max() > 1:
	img = img/277			

if args.rows and args.cols:
	img = resize(img, (int(args.rows),int(args.cols)))
else:
	img = resize(img, (512,512))
shape = img.shape
img = np.reshape(img, (1, shape[0],shape[1], shape[2]))

#Load de model and weigths to predict the output
model = load_model(model_path, custom_objects={'loss_mix_v3': loss_definition.loss_mix_v3})
model.load_weights(weights_path)
out = np.clip(model.predict(img), .0, 1.)

if args.output:
	for o in out:
		plt.imshow(o)
		plt.imsave(str(args.output)+'.png', o)
else:
	for o in out:
		plt.imshow(o)
		plt.imsave('output.png', o)
