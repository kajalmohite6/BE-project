import keras
import keras.utils
from keras import utils as np_utils
from keras.utils.np_utils import to_categorical # newly added by me

#from keras.preprocessing.image import ImageDataGenerators
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage import io
from keras.preprocessing import image
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib
from keras.optimizers import SGD
import os
import cv2

#import os


img_width, img_height = 28, 28
nb_train_samples = 14528            #neverUsed
nb_validation_samples = 3532        #neverUsed
epochs = 200
batch_size = 16


def load_data(data_directory):
    
    directories = [d for d in os.listdir(data_directory) 
    
                   if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    
    images = []
    
    for d in directories:
        
        label_directory = os.path.join(data_directory, d)
        
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      ]
        
        for f in file_names:
            
            images.append(transform.resize(skimage.data.imread(f),(img_width, img_height)))
            labels.append(int(d))
            
    return images, labels

train_data_dir = '/home/research_centre_gpu/train'
validation_data_dir = '/home/research_centre_gpu/test'

    
images, labels = load_data(train_data_dir)
val_images, val_labels = load_data(validation_data_dir)

images_64=np.array(images).astype(np.float32) 
labels_64=np.array(labels)
val_images_64=np.array(val_images).astype(np.float32) #uint8 newly added
val_labels_64=np.array(val_labels)

io.imshow(images_64[1050])
io.imshow(val_images_64[1050]) 

#images_64 = images_64/255
#val_images_64 = val_images_64/255

#pt='/home/bhagu/Documents/sohail/data/segmented/training_set/1/8c1a0a73-26c9-405e-8957-8dd58df6bdf4___GCREC_Bact.Sp 3738_final_masked.jpg'
#test_img=data.imread(pt)
#ri=test_img * 255
#test_img2=np.array(test_img).astype(np.uint8)
#
#test_img2 = test_img2/255
#print(test_img2[10])
##val_images_64 = val_images_64/255
#io.imshow(test_img)
#io.imshow(test_img2)
#plt.imshow(test_img2)
#print(images_64[0])

print(images_64.shape)
print(labels_64.shape)
print(val_images_64.shape) 
print(val_labels_64.shape)

#train_images=images_64.reshape(14528,img_height*img_width*3).astype('float16')
#val_images=val_images_64.reshape(3532,img_height*img_width*3).astype('float16')

train_labels=to_categorical(labels_64) #
val_labels=to_categorical(val_labels_64) # newly

print(train_labels.shape)
print(val_labels.shape)
#final_lbls=labels_64.reshape((10))
#images_64_gr=rgb2gray(images_64)
#val_images_64_gr=rgb2gray(val_images_64)

#io.imshow(images_64_gr[0])

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_height,img_width,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(38))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x=images_64,
          y=train_labels,
          validation_data=(val_images_64,val_labels),
          epochs=epochs,
          batch_size=16
          )


model.save('color_28_all_v1.h5')	

