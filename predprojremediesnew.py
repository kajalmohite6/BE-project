
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import load_model
#from keras.preprocessing import image
from keras.utils import to_categorical
import skimage
from skimage import data
from skimage import transform
#from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import base64
import numpy as np
import io
from io import BytesIO
from PIL import Image
import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
#model=load_model('/home/kajal/beprojfinal/color_28_all_v1.h5')


app=Flask(__name__)
def get_model():
    global model
    model=load_model('/home/kajal/beprojfinal/color_28_all_v1.h5')
    print("*Model Loaded")

#model=load_model('/home/kajal/Downloads/color_28_all_v1_NEW_.h5')




print("*Loading Keras model...")
get_model()
@app.route('/proj4',methods=['GET','POST'])
def proj3():
    # print("IHI")
    message=request.get_json(force=True)

    encoded=message['image'].split(",")[1]
    imgdata = base64.b64decode(encoded)
    filename = 'some_image.JPG'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    #decoded.seek(0)
    #ded=decoded.read();
    #byteImgIO = io.BytesIO()
    #byteImg = Image.open("some/location/to/a/file/in/my/directories.png")
    #decoded.save(byteImgIO, "JPG")
    #byteImgIO.seek(0)
    #decoded = byteImgIO.read()
    img1=skimage.transform.resize(skimage.data.imread(filename),(28,28))
    # print("IHI3")
    try:
        img=np.array(img1,dtype=np.float32).reshape((1,28,28,3))
    # print("IHI4")
        a=model.predict_classes(img)
    # print("IHI2")
        prediction_lab=a[0]
        print(prediction_lab)
        reverse_mapping =  ['Apple_apple_scab','Apple_black_rot','Apple_cedar_apple_rust','Apple_healthy','Blueberry_healthy',
        'Cherry_powder_mildew','Cherry_healthy','corn_maize_leafspot','corn_common_rust','corn_leaf_blight','corn_healthy',
        'Grape_Black_rot','Grape_Esca','Grape_leafblight','Grape_healthy','Orange_CitrusGreening','Peech_Bacterialspot','Peech_healthy',
        'Pepper_bell_Bacterialspot','Pepperbell_healthy','Potato Early Blight','Potato late blight','Potato Healthy','Raspberry_healthy',
        'Soyabean Healthy','Squash Powdery Mildew','Strawberry leaf scorch','Strawberry healthy','Tomato bacterial spot',
        'Tomato Early Blight','Tomato Yellow Leaf Curl virus','Tomato healthy','Tomato Late Blight','Tomato Leaf Mold',
        'Tomato septoria leaf spot','Tomato Two Spotted Spider Mite','Tomato mosaic virus','Tomato Target Spot' ]



        reverse_mapping_rem=['Containing sulfur and pyrethrins, Bonide® Orchard Spray is a safe one-hit concentrate for 		insectattacks',
        	'Prune out cankers, dead branches, twigs, etc.',
		'Choose resistant cultivators and dispose of fallen leaves and other debris from under trees.',
		'none',
		'none',
		'Choose resistant cultivators and dispose of fallen leaves and other debris from under trees.',
		'none',
		' use resistant plant varieties and Spray fungicides early in season before initial damage.',
		'use resistant plant varieties and Spray fungicides early in season before initial damage.',
		'none',
		'Sanitation is must. Destroy mummies, remove diseased tendrils from the wires, and select fruiting canes 			 without lesions.',
		'Sanitation is must. Destroy mummies, remove diseased tendrils from the wires, and select fruiting canes 			 without lesions.',
		'Sanitation is must. Destroy mummies, remove diseased tendrils from the wires, and select fruiting canes 			 without lesions.',
		'none','cultural methods such as antibacterial management, sanitation, removal of infected plants,frequent 			 scouting can cure the disease effectively.',
		'Copper works very well keeping fruit clean of bacterial spot and does not cause fruit damage, like fruit 			 russeting observed on apples when copper is used post dormancy.',
		'none',
		'Copper works very well keeping fruit clean of bacterial spot and does not cause fruit damage, like fruit 			 russeting observed on apples when copper is used post dormancy.',
		'none',
		'Copper works very well keeping fruit clean of bacterial spot and does not cause fruit damage, like fruit 			 russeting observed on apples when copper is used post dormancy.',
		'Copper works very well keeping fruit clean of bacterial spot and does not cause fruit damage, like fruit 		         russeting observed on apples when copper is used post dormancy.',
		'none',
		'none',
		'none',
		'Baking Soda (sodium bicarbonate)-This is possibly the best known of the home-made, organic solutions for 			 powdery mildew.Although studies indicate that baking soda alone is not all that effective, when combined 			 with horticultural grade or dormant oil and liquid soap, efficacy is very good if applied in the early 		 stages or before an outbreak occurs.',
		'Containing sulfur and pyrethrins, Bonide® Orchard Spray is a safe, one-hit concentrate for insect attacks 			 and fungal problems.',
		'none',
		'To avoid bacterial spot, cultivators should buy certified disease-free tomato seeds and use sterilized soil 			 or a mix that is commercially rendered.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.',
		'none',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plants germination.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.',
		'If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 			1.3% sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in 			submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed 			bacteria, but might adversely affect the plant’s germination.'
]

    # In[26]:


        prediction = reverse_mapping[prediction_lab]
        remed = reverse_mapping_rem[prediction_lab]
        print(prediction)
    #processed_image=preprocess_image(image,target_size=(224,224))
    #prediction=model.predict(processed_image).tolist()
        response={
          'prediction':{
              'leaf':prediction,
              'remedies':remed
           }
        }

        print(response)
    #except Exception as e:
        return jsonify(response)

    except Exception as e:
        response={
          'prediction':{
              'leaf':"Wrong Input"
          #'cat':prediction[0][1]
           }
        }

        print(response)
    #except Exception as e:
        return jsonify(response)


if __name__ == '__main__':
    app.run()

