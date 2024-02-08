import numpy as np
import tensorflow

np.random.seed(1) # compare with same random seed between 2 different accelerators

tensorflow.keras.backend.set_image_data_format('channels_last') # force dimension NHWC

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

# generate randon image
x =  np.random.randint(0,256,(1, 224,224,3))
x = preprocess_input(x)

preds = model.predict(x)

# Let s compare differemt predictions between different devices
print('Predicted:', decode_predictions(preds, top=3)[0])

