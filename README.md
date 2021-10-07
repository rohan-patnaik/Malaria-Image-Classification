# Malaria-Image-Classification
Image classification used on a kaggle malaria dataset using CNN, ResNet50, VGG19 and InceptionV3.

Follow to see the end to end deployment using heroku for this soon!

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

# Time to compile the model
model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"]
)

# The below written code is if you want to create a model from scratch

from tensorflow.keras.layers import MaxPool2D\
model=Sequential()\
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))\
model.add(MaxPooling2D(pool_size=2))\
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))\
model.add(MaxPooling2D(pool_size=2))\
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))\
model.add(MaxPooling2D(pool_size=2))\
model.add(Flatten())\
model.add(Dense(500,activation="relu"))\
model.add(Dense(2,activation="softmax"))\
model.summary()

# Use the Image Data Generator to import images from the dataset
train_datagen = ImageDataGenerator(rescale=1./255,\
                                   shear_range=0.2,\
                                   zoom_range=0.2,\
                                   horizontal_flip=True)\
test_datagen = ImageDataGenerator(rescale=1./255)\

# Make sure you provide the same target size as initialised for the image

training_set = train_datagen.flow_from_directory("DS1/Dataset/Train/",
                                                target_size=(224,224),
                                                batch_size=2,
                                                class_mode="categorical")

test_set = test_datagen.flow_from_directory("DS1/Dataset/Test/",
                                           target_size=(224,224),
                                           batch_size=2,
                                           class_mode="categorical")

# Now we fit the model
hist = model.fit_generator(training_set,
                 validation_data=test_set,
                 epochs=2,
                 steps_per_epoch=len(training_set),
                 validation_steps=len(test_set)
)

# Now lets plot the loss and accuracy for the model we just created
#loss
plt.plot(hist.history['loss'], label='train loss')\
plt.plot(hist.history['val_loss'], label='val loss')\
plt.legend()\
plt.show()\
plt.savefig('LossVal_loss')

#accuracies
plt.plot(hist.history['accuracy'], label='train acc')\
plt.plot(hist.history['val_accuracy'], label='val acc')\
plt.legend()\
plt.show()\
plt.savefig('AccVal_acc')

hist.history\
model.save('model_vgg19.h5')\

y_pred = model.predict(test_set)\

y_preds = np.argmax(y_pred, axis=1)\
y_preds\

# Now we read the image and predict the outcome of the image given

from tensorflow.keras.models import load_model\
model = load_model("model_vgg19.h5")

# For checking a specific image falling under a category
Go ahead and import it.
Then convert the image to an array for the computer to process it.
Just to make sure check its shape so its dimensions are in accordance to what we need.
Now rescale the image since this image is not scaled as we did for train_datagen and test_datagen.
Next go ahead and expand its dimensions for it to be of the same shape.
Now we apply the predict function on this image using the model we created.
All we need to do now is to check which category it falls into. 

# The snippets below will give you a brief idea of how to proceed
from tensorflow.keras.preprocessing import image
img=image.load_img('DS1/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_105803_cell_79.png',target_size=(224,224))

x = image.img_to_array(img)

x.shape

x = x/255

x = np.expand_dims(x, axis=0)\
img_data = preprocess_input(x)\
img_data.shape

model.predict(img_data)

a=np.argmax(model.predict(img_data), axis=1)

if (a==1):
    print("Uninfected")
else:
    print("Infected")
