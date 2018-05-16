import keras as K
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
import numpy as np

# img_path = 'Silvio_Berlusconi_0023.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# y = image.img_to_array(img)
# print(y.shape)
# y = np.expand_dims(y, axis=0)
# print(y.shape)
# y = preprocess_input(y)
# print(y.shape)

class MMobileNet():

	base_model = None

	def __init__(self):
		self.base_model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, 
										include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

	def __call__(self, x, layers=['conv1', 'conv_pw_5_bn']):
		x = np.expand_dims(x, axis=0)
		model = Model(inputs=self.base_model.input, outputs=[self.base_model.get_layer(layer).output for layer in layers])
		features = model.predict(x)
		return features

# test = mobileNet()
# layers2 = test(y)
# for i in layers2:
# 	print(i.shape)