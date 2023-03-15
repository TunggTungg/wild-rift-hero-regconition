from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

def load_embb_model(input_size):
	model = Sequential()
	# add the pretrained model
	model.add(VGG19(input_shape=(input_size,input_size,3), include_top = False, weights = 'imagenet'))
	model.add(Flatten())
	model.layers[0].trainable=False

	return model