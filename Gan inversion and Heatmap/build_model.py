from prepare_data import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
import tensorflow_addons as tfa
from loss import *
import time


class Res_block(Model):
    def __init__(self,output_plain):
        super(Res_block, self).__init__()
        self.conv1 = Conv2D(output_plain, kernel_size=3, padding='same',activation="relu", kernel_initializer='glorot_normal')
        self.BN2 = BatchNormalization()
        self.conv2 = Conv2D(output_plain, kernel_size=3, padding='same',activation="relu",kernel_initializer='glorot_normal')
        self.In = tfa.layers.InstanceNormalization()

    def call(self, inputs, training=False, **kwargs):
        res = self.conv1(inputs)
        out = self.In(res)
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
    out = Flatten()(out)  #out = Flatten()(inputs)
    out = Dense(512, activation=LeakyReLU(0.4))(out)  # ck_data 2048
    out = Dropout(0.4)(out)
    out = Dense(256, activation=LeakyReLU(0.4))(out)  # ck_data 256
    out = Dropout(0.4)(out)
    out = Dense(200, activation=LeakyReLU(0.4))(out)
    encoder = Model(inputs,out)
    return encoder


def generator():
    input = Input((200))
    d1 = Dense(4 * 4 * 1024, use_bias=False, name="d1")(input)
    bat1 = BatchNormalization(name="bat1")(d1)
    ac1 = LeakyReLU(0.3, name="ac1")(bat1)
    reshape = Reshape((4, 4, 1024), name="reshape")(ac1)
    deconv1 = Conv2DTranspose(512, 5, strides=2, padding="same", use_bias=False, name="deconv1")(reshape)
    bat2 = tfa.layers.InstanceNormalization()(deconv1)
    ac2 = LeakyReLU(0.3, name="ac2")(bat2)
    deconv2 = Conv2DTranspose(256, 5, strides=2, padding="same", use_bias=False, name="deocnv2")(ac2)
    bat3 = tfa.layers.InstanceNormalization()(deconv2)
    ac3 = LeakyReLU(0.3, name="ac3")(bat3)
    deconv3 = Conv2DTranspose(128, 5, strides=2, padding="same", use_bias=False, name="deconv3")(ac3)
    bat4 = tfa.layers.InstanceNormalization()(deconv3)
    ac4 = LeakyReLU(0.3, name="ac4")(bat4)
    output = Conv2DTranspose(1, 5, strides=2, padding="same", use_bias=False, activation="tanh", name="output")(ac4)
    model = Model(input, output)
    model.summary()
    return model


#discriminator encoder
def discriminator():
    input = Input((64, 64, 1))
    # 32 32
    out = Conv2D(64, 5, activation=LeakyReLU(0.3), strides=2, padding="same", name="conv1")(input)
    out = Dropout(0.3, name="drop1")(out)
    # 16 16
    out = Conv2D(128, 5, activation=LeakyReLU(0.3), strides=2, padding="same", name="conv2")(out)
    out = Dropout(0.3, name="drop2")(out)
    out = Flatten(name="flat")(out)
    output = Dense(1, activation="sigmoid", name="output")(out)
    model = Model(input, output)
    model.summary()
    return model


def discriminator_decoder():
    inputs = Input((1))
    out = Dense(32, activation=LeakyReLU(0.3))(inputs)
    out = Dropout(0.3)(out)
    out = Dense(1024, activation=LeakyReLU(0.3))(out)
    out = Dropout(0.3)(out)
    out = Dense(8 * 8 * 1024, activation=LeakyReLU(0.3))(out)
    out = Reshape((8, 8, 1024))(out)
    out = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(out)
    out = Conv2D(128,3, activation=LeakyReLU(0.3),padding="same")(out)
    out = Conv2D(128,3, activation=LeakyReLU(0.3), padding="same")(out)
    out = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(out)
    out = Conv2D(64, 3, activation=LeakyReLU(0.3), padding="same")(out)
    out = Conv2D(64, 3, activation="softmax", padding="same")(out)
    out = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(out)
    out = Conv2D(64, 3, activation=LeakyReLU(0.3), padding="same")(out)
    out = Conv2D(2, 3, activation="softmax", padding="same")(out)
    model = Model(inputs, out)
    return model


# def discriminator_decoder():
#     model = discriminator()
#     model.load_weights("model_weight/instance_discriminator/discriminator-500")
#     for layer in model.layers:
#         layer.trainable = False
#
#     network_layer_name = ["conv1", "drop1", "conv2", "drop2", "flat", "output"]
#     inputs = Input((64,64,1))
#     output1 = model.get_layer("conv1")(inputs)
#     output2, output_global = inputs, inputs
#     for layer in network_layer_name[0:3]:
#         output2 = model.get_layer(layer)(output2)
#     for layer in network_layer_name:
#         output_global = model.get_layer(layer)(output_global)
#
#     out = Dense(32,activation=LeakyReLU(0.3))(output_global)
#     out = Dropout(0.3)(out)
#     out = Dense(1024, activation=LeakyReLU(0.3))(out)
#     out = Dropout(0.3)(out)
#     out = Dense(8*8*1024, activation=LeakyReLU(0.3))(out)
#     out = Reshape((8,8,1024))(out)
#     out = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(out)
#     merge = tf.concat((out, output2), axis=3)
#     out = tfa.layers.InstanceNormalization()(merge)
#     out = Conv2D(64, 3, activation=LeakyReLU(0.3), strides=1, padding="same", )(out)
#     out = Conv2D(64, 3, activation=LeakyReLU(0.3), strides=1, padding="same", )(out)

    # out = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(out)
    # merge = tf.concat((out, output1), axis=3)
    # out = tfa.layers.InstanceNormalization()(merge)
    # out = Conv2D(128, 3, activation=LeakyReLU(0.3), strides=1, padding="same", )(out)
    # out = Conv2D(64, 3, activation=LeakyReLU(0.3), strides=1, padding="same", )(out)

    # out = Conv2DTranspose(2, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(out)
    # merge = tf.concat((out, inputs), axis=3)
    # out = tfa.layers.InstanceNormalization()(merge)
    # out = Conv2D(2, 3, activation="softmax", strides=1, padding="same", )(out)

    # model = Model(inputs,  out)
    # model.summary()
    # return model
#
# model = discriminator_decoder()


def mapping_network():
    inputs = Input((200))
    res = Dense(200,activation=LeakyReLU(0.5),kernel_initializer="normal")(inputs)
    out = Dropout(0.4)(res)
    out = Dense(200, activation=LeakyReLU(0.5), kernel_initializer="normal")(out)
    out = Dropout(0.4)(out)
    # out = res + out

    res = Dense(200, activation=LeakyReLU(0.5), kernel_initializer="normal")(out)
    out = Dropout(0.4)(res)
    out = Dense(200, activation=LeakyReLU(0.5), kernel_initializer="normal")(out)
    out = Dropout(0.4)(out)
    # out = res + out

    res = Dense(200,activation=LeakyReLU(0.5),kernel_initializer="normal")(out)
    out = Dropout(0.4)(res)
    out = Dense(200, activation=LeakyReLU(0.5), kernel_initializer="normal")(out)
    out = Dropout(0.4)(out)
    # out = res + out


    model = Model(inputs, out)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(2e-4)
    model.compile(optimizer=optimizer, loss="mse")
    return model

