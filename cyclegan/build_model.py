import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

class Resblock(Layer):
    def __init__(self, chn, strides):
        super(Resblock, self).__init__()
        self.chn = chn
        self.strides = strides
        self.conv1 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=LeakyReLU(0.3))
        self.bn = BatchNormalization()
        self.conv2 = Conv2D(self.chn, 3, padding='same', activation=LeakyReLU(0.3))
        self.conv3 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=None)

    def __call__(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        if inputs.shape[-1] == self.chn:
            return tf.math.add(inputs, conv2)
        else:
            conv3 = self.conv3(inputs)
            return tf.math.add(conv2, conv3)

def build_generator(input_shape=(128,128,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 5, strides=(1, 1), padding='same', activation=LeakyReLU(0.3))(inputs)
    bn = BatchNormalization()(conv1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(conv2)
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(conv3)
    for i in range(6):
        resblock = Resblock(256, strides=(1, 1))(bn)
        bn = BatchNormalization()(resblock)

    dconv1 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    print(dconv1.shape)
    bn = BatchNormalization()(dconv1)
    dconv2 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(dconv2)
    outputs = Conv2D(1, 5, strides=(1, 1), padding='same', activation='tanh')(bn)

    model = Model(inputs, outputs)
    model.summary()

    return model

def build_discriminator(input_shape=(128,128,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.1))(inputs)  # 28 -> 14
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.1))(conv1)  # 14 -> 7
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.1))(conv2)  # 7 -> 3
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.1))(conv3)
    flat = Flatten()(conv4)
    d1 = Dense(128,activation=LeakyReLU(0.3))(flat)
    d2 = Dense(64,activation=LeakyReLU(0.3))(d1)
    output = Dense(1, activation="softmax")(d2)
    # validation = Conv2D(64, 3, strides=(2, 2), padding='same')(conv4)
    # validation = Conv2D(1, 3, strides=(2, 2), padding='same', activation="sigmoid")(validation)
    # validation = Softmax(axis=-1)(validation)
    model = Model(inputs, output)
    model.summary()
    return model




