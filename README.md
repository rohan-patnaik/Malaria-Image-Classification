# Malaria-Image-Classification
Image classification used on a kaggle malaria dataset using CNN, ResNet50, VGG19 and InceptionV3.

Follow to see the end to end deployment using heroku for this as well!

Download the dataset from the link below
https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

# Change the file directories and image paths accordingly when using in kaggle

#Lets start by importing the basic libraries first of all as shown below\
import tensorflow as tf\
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D\
from tensorflow.keras.models import Model\
from tensorflow.keras.applications.vgg19 import VGG19\
from tensorflow.keras.applications.resnet50 import preprocess_input\
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img\
from tensorflow.keras.models import Sequential\
import numpy as np\
import pandas as pd\
from glob import glob\
import matplotlib.pyplot as plt\
%matplotlib inline

# Now we can resize the images to a standard

IMAGE_SIZE = [224, 224]

# Add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

#Something I found online as a reasoning for the layer.trainable = False\
It is very common in transfer learning to completely freeze the transferred layers in order to preserve them. In the early stages of training your additional layers don't know what to do. That means a noisy gradient by the time it gets to the transferred layers, which will quickly "detune" them away from their previously well-tuned weights.

# Useful for getting number of classes
folders = glob('DS1/Dataset/Train/*')
folders

#layers - you can add more if you want
x = Flatten()(vgg.output)\
#x = Dense(1000, activation='relu')(x)\
prediction = Dense(len(folders), activation='softmax')(x)

# Now we create the base model and check for an overall summary(Remembber to CCFE - create, compile, fit and evaluate)
model = Model(inputs = vgg.input, outputs = prediction)\
model.summary()










