from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Concatenate, UpSampling2D, Conv2DTranspose
from tensorflow.keras.initializer import RandomNormal
import tensorflow as tf

def Generator(img_size=256):
	input = Input(shape=[img_size, img_size, 1])
	initializer = RandomNormal(stddev=0.02)

	#Encoder
	#conv layer1
	conv1 = Conv2D(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(input)
	conv1 = ReLU()(conv1)

	#conv layer2
	conv2 = Conv2D(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(conv1)
	conv2 = shortcut_deconv8 = ReLU()(conv2)

	#conv layer3
	conv3 = Conv2D(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(conv2)
	conv3 = ReLU()(conv3)

	#conv layer4 
	conv4 = Conv2D(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(conv3)
	conv4 = shortcut_deconv6 = ReLU()(conv4)

	#conv layer5
	conv5 = Conv2D(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(conv4)
	conv5 = ReLU()(conv5)


	#Decoder
	#deconv layer 6 
	deconv6 = Conv2DTranspose(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(conv5)
	deconv6 += shortcut_deconv6
	deconv6 = ReLU()(deconv6)

	#deconv7
	deconv7 = Conv2DTranspose(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(deconv6)
	deconv7 = ReLU()(deconv7)

	#deconv 8 
	deconv8 = Conv2DTranspose(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(deconv7)
	deconv8 += shortcut_deconv8
	deconv8 = ReLU()(deconv8)

	#deconv9
	deconv9 = Conv2DTranspose(96, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(deconv8)
	deconv9 = ReLU()(deconv9)

	#deconv 10 
	deconv10 = Conv2DTranspose(1, kernel_size=5, strides=1, padding='valid', kernel_initializer=initializer)(deconv9)
	deconv10 += input

	#output
	output = ReLU()(deconv10)

	return Model(inputs=input, outputs=output)

