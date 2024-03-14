'''
from load_data import *
import tensorflow as tf
from build_model import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from loss import *
import time


class Res_block(Model):
    def __init__(self,output_plain):
        super(Res_block, self).__init__()
        self.conv1 = Conv2D(output_plain, kernel_size=3, padding='same',activation="relu", kernel_initializer='glorot_normal')
        self.BN2 = BatchNormalization()
        self.conv2 = Conv2D(output_plain, kernel_size=3, padding='same',activation="relu",kernel_initializer='glorot_normal')


    def call(self, inputs, training=False, **kwargs):
        res = self.conv1(inputs)
        out = self.BN2(res)
        out = self.conv2(out)
        res += out
        return res

def encoder():
    inputs = Input((64,64,1))
    out = Res_block(64)(inputs)
    out = Res_block(64)(out)
    out = Res_block(64)(out)
    out = Res_block(1)(out)
    out = out + inputs
    out = Flatten()(inputs)  #out = Flatten()(inputs)
    out = Dense(512, activation=LeakyReLU(0.4))(out)  # ck_data 2048
    out = Dropout(0.4)(out)
    out = Dense(256, activation=LeakyReLU(0.4))(out)  # ck_data 256
    out = Dropout(0.4)(out)
    out = Dense(200, activation=LeakyReLU(0.4))(out)
    encoder = Model(inputs,out)
    return encoder


def generator():
    input = Input((100))
    d1 = Dense(4*4*1024,use_bias=False,name="d1")(input)
    bat1 = BatchNormalization(name="bat1")(d1)
    ac1 = LeakyReLU(0.3,name="ac1")(bat1)
    reshape = Reshape((4,4,1024),name="reshape")(ac1)
    deconv1 = Conv2DTranspose(512,5,strides=2,padding="same",use_bias=False,name="deconv1")(reshape)
    bat2 = BatchNormalization(name="bat2")(deconv1)
    ac2 = LeakyReLU(0.3,name="ac2")(bat2)
    deconv2 = Conv2DTranspose(256,5,strides=2,padding="same",use_bias=False,name="deocnv2")(ac2)
    bat3 = BatchNormalization(name="bat3")(deconv2)
    ac3 = LeakyReLU(0.3,name="ac3")(bat3)
    deconv3 = Conv2DTranspose(128,5,strides=2,padding="same",use_bias=False,name="deconv3")(ac3)
    bat4 = BatchNormalization(name="bat4")(deconv3)
    ac4 = LeakyReLU(0.3,name="ac4")(bat4)
    output = Conv2DTranspose(1,5,strides=2,padding="same",use_bias=False,activation="tanh",name="output")(ac4)
    model = Model(input,output,name="generator")
    model.summary()
    return model


def discriminator():
    input = Input((64,64,1))
    conv1 = Conv2D(64,5,activation=LeakyReLU(0.3),strides=2,padding="same",name="conv1")(input)
    drop1 = Dropout(0.3,name="drop1")(conv1)
    conv2 = Conv2D(128,5,activation=LeakyReLU(0.3),strides=2,padding="same",name="conv2")(drop1)
    drop2 = Dropout(0.3,name="drop2")(conv2)
    flat = Flatten(name="flat")(drop2)
    output = Dense(1,name="output")(flat)
    model = Model(input,output,name="discriminator")
    model.summary()
    return model

if __name__ == "__main__":
    generator = generator()
    discriminator = discriminator()
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def create_generator():
    input = Input((100))
    d1 = Dense(4*4*1024,use_bias=False,name="d1")(input)
    bat1 = BatchNormalization(name="bat1")(d1)
    ac1 = LeakyReLU(0.3,name="ac1")(bat1)
    reshape = Reshape((4,4,1024),name="reshape")(ac1)
    deconv1 = Conv2DTranspose(512,5,strides=2,padding="same",use_bias=False,name="deconv1")(reshape)
    bat2 = BatchNormalization(name="bat2")(deconv1)
    ac2 = LeakyReLU(0.3,name="ac2")(bat2)
    deconv2 = Conv2DTranspose(256,5,strides=2,padding="same",use_bias=False,name="deocnv2")(ac2)
    bat3 = BatchNormalization(name="bat3")(deconv2)
    ac3 = LeakyReLU(0.3,name="ac3")(bat3)
    deconv3 = Conv2DTranspose(128,5,strides=2,padding="same",use_bias=False,name="deconv3")(ac3)
    bat4 = BatchNormalization(name="bat4")(deconv3)
    ac4 = LeakyReLU(0.3,name="ac4")(bat4)
    output = Conv2DTranspose(1,5,strides=2,padding="same",use_bias=False,activation="tanh",name="output")(ac4)
    model = Model(input,output,name="generator")
    model.summary()
    return model


def create_discriminator():
    input = Input((64,64,1))
    conv1 = Conv2D(64,5,activation=LeakyReLU(0.3),strides=2,padding="same",name="conv1")(input)
    drop1 = Dropout(0.3,name="drop1")(conv1)
    conv2 = Conv2D(128,5,activation=LeakyReLU(0.3),strides=2,padding="same",name="conv2")(drop1)
    drop2 = Dropout(0.3,name="drop2")(conv2)
    flat = Flatten(name="flat")(drop2)
    output = Dense(1, name="output", activation="sigmoid")(flat)
    model = Model(input, output, name="discriminator")
    model.summary()
    return model

def create_discriminator_patch_gan():
    input = Input((64,64,1))
    conv1 = Conv2D(32,(3,3),strides=(2,2),padding="same",activation=LeakyReLU(0.01),name="conv1")(input)
    conv2 = Conv2D(64,(3,3),strides=(2,2),padding="same",activation=LeakyReLU(0.01),name="conv2")(conv1)
    conv3 = Conv2D(128,(3,3),strides=(2,2),padding="same",activation=LeakyReLU(0.01),name="conv3")(conv2)
    conv4 = Conv2D(256,(3,3),strides=(2,2),padding="same",activation=LeakyReLU(0.01),name="conv4")(conv3)
    validation = Conv2D(64,(3,3),strides=(2,2),padding="same",activation=LeakyReLU(0.01),name="validation")(conv4)
    output = Conv2D(2,(3,3),strides=(1,1),padding="same",activation="softmax",name='output')(validation)
    model = Model(input,output,name="patchGAN_discriminator")
    model.summary()
    return model



if __name__ == "__main__":
    generator = create_generator()
    discriminator = create_discriminator()





    # noise = tf.random.normal([1,100])
    # generated_image = generator(noise)
    # generated_image = np.array(generated_image)
    # print(generated_image[0,:,:,0].shape)
    # plt.imshow(generated_image[0,:,:,0])
    # plt.show()
