from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical
import skimage
from skimage import data
from skimage import transform
from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#model=load_model('/home/kajal/beprojfinal/color_28_all_v1.h5')

model=load_model('/home/kajal/Downloads/color_28_all_v1_NEW_.h5')

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    fname = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      ]
        for f in file_names:
            images.append(skimage.transform.resize(skimage.data.imread(f),(28,28)))
            labels.append(int(d))
            fname.append(f)
    return images, labels, fname



test_data_directory = "/home/kajal/test"

images, labels, fname = load_data(test_data_directory)
#print(len(images))
#print(len(labels))
#print(len(fname))
images_64=np.array(images,dtype=np.float32)
#print(images_64)
labels_64=np.array(labels)
#print(labels_64)
#print(images_64.dtype)
#print(images_64.shape)
#print(labels_64.shape)

#test_images=images_64.reshape(100,64*64*3).astype('float32')

test_labels=to_categorical(labels_64)
#print(test_labels)
#print(test_images.shape)
#print(test_labels.shape)
#print(labels)
model.evaluate(x=images_64,y=test_labels)
pred=model.predict_classes(images_64)
#print(pred)
#img_path="/home/research_centre_gpu/test/2/1ac2fcfb-92ad-4fda-9aa7-7e4450af6647.JPG"
img_path2="/home/kajal/cat/orange.JPG"
img1=skimage.transform.resize(skimage.data.imread(img_path2),(28,28))
img=np.array(img1,dtype=np.float32).reshape((1,28,28,3))
#io.imshow(img)
#img1=np.array(img1).reshape((1,64,64,3)).astype('float32')
#print(img1.dtype)
#x=np.array(img1)
#print(img.shape)
a=model.predict_classes(img)
prediction=a[0]
print(prediction)


reverse_mapping = ['Apple_apple_scab','Apple_black_rot','Apple_cedar_apple_rust','Apple_healthy','Blueberry_healthy','Cherry_powder_mildew',
'Cherry_healthy','corn_maize_leafspot','corn_common_rust','corn_leaf_blight','corn_healthy','Grape_Black_rot','Grape_Esca',
'Grape_leafblight','Grape_healthy','Orange_CitrusGreening','Peech_Bacterialspot','Peech_healthy','Pepper_bell_Bacterialspot','Pepperbell_healthy',
'Potato Early Blight','Potato late blight','Potato Healthy','Raspberry_healthy','Soyabean Healthy','Squash Powdery Mildew','Strawberry leaf scorch','Strawberry healthy','Tomato bacterial spot','Tomato Early Blight','Tomato Yellow Leaf Curl virus','Tomato healthy','Tomato Late Blight','Tomato Leaf Mold','Tomato septoria leaf spot','Tomato Two Spotted Spider Mite','Tomato mosaic virus','Tomato Target Spot' ]


# In[26]:


prediction_name = reverse_mapping[prediction]
print(prediction_name)
#reverse_mapping = [' Corn Northern Leaf Blight','Potato Late Blight','Corn Common Rust','Corn Gray Leaf Spot', 'Corn healthy','Potato Early Blight','Potato Healthy','Tomato Early Blight','Tomato Late Blight','Tomato Leaf Mold','Tomato Target Spot','Tomato Yellow Leaf Curl','Tomato Bacterial Spot','Tomato Two Spotted Spider Mite','Tomato healthy','grape black measles','grape black rot','grape healthy', 'grape leaf blight' ]
#print(labels)
#print(pred)
#for i in range (len(labels)):
 #   if labels[i]!=pred[i]:
  #     print(fname[i])
   #if labels[i]==pred[i]:
     # print(labels[i],fname[i])
   
    #else:
     # print(pred[i])
#lbl=lbl.reshape(10)
#encoded = to_categorical(labels_64)
#
#print(len(encoded))
#print(images_64.shape)
#print(labels_64.shape)
#
#test_data_dir = '/home/bhagu/Documents/sohail/data/color/test_set'
#test_datagen = ImageDataGenerator()
#test_generator = test_datagen.flow_from_directory(
#        test_data_dir,
#        target_size=(64,64),
#        class_mode='categorical'
#        )
#
#x,y=test_generator.next()
#print(len(y[0]))
#print(len(test_generator.classes.shape))
#model.evaluate(x=images_64,y=encoded)
#img_path="/home/bhagu/Documents/sohail/data/color/training_set/10/Yellow_Leaf_curl_virus_1084.JPG"
#
#img1=image.load_img(img_path,target_size=(64,64))
#img1=np.array(img1).reshape((1,64,64,3))
#x=np.array(img1)
#print(x.shape)
#print(len(y))
#print(test_generator.class_indices)
#model.predict_classes(x)
