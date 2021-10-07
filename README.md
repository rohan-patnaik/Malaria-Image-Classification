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

#Now we can resize the images to a standard\
IMAGE_SIZE = [224, 224]
