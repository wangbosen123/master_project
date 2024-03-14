import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
import numpy as np


class res_block(Model):
    def __init__(self,output_plain):
        super(res_block, self).__init__()
        self.conv1 = Conv2D(output_plain, kernel_size=3, padding='same',activation=LeakyReLU(0.3), kernel_initializer='glorot_normal')
        self.conv2 = Conv2D(output_plain, kernel_size=3, padding='same',activation=LeakyReLU(0.3),  kernel_initializer='glorot_normal')
        self.conv3 = Conv2D(output_plain, kernel_size=3, padding='same',activation=LeakyReLU(0.3),  kernel_initializer='glorot_normal')

        self.In1 = tfa.layers.InstanceNormalization()
        self.In2 = tfa.layers.InstanceNormalization()
        # self.In3 = tfa.layers.InstanceNormalization()


    def call(self, inputs, training=True, **kwargs):
        res = self.conv1(inputs)
        out = self.In1(res)
        out = self.conv2(out)
        out = self.In2(out)
        res += out
        return res

class residual_block_up(Model):
    def __init__(self, channels):
        super(residual_block_up, self).__init__()
        initializer = tf.keras.initializers.Orthogonal()
        #block a
        self.up1 = UpSampling2D()
        self.conv1 = Conv2D(channels, 1, activation=LeakyReLU(0.3),kernel_initializer=initializer)
        self.In1 = tfa.layers.InstanceNormalization()

        #block b
        self.up2 = UpSampling2D()
        self.In2 = tfa.layers.InstanceNormalization()
        self.conv2 = Conv2D(channels, 3, padding='same', activation=LeakyReLU(0.3), kernel_initializer=initializer)

    def call(self, inputs):
        #block a
        x = self.up1(inputs)
        x = self.conv1(x)
        output1 = self.In1(x)

        #block b
        x = self.conv2(inputs)
        x = self.In2(x)
        output2 = self.up2(x)
        return output1 + output2

class residual_block_down(Model):
    def __init__(self, channels, down=True):
        super(residual_block_down, self).__init__()
        initializer = tf.keras.initializers.Orthogonal()
        self.down = down
        self.conv1 = Conv2D(channels, 1, activation=LeakyReLU(0.3),  kernel_initializer=initializer)
        self.In1 = tfa.layers.InstanceNormalization()
        if down:
            self.AVP1 = AveragePooling2D()
            self.AVP2 = AveragePooling2D()
        self.conv2 = Conv2D(channels, 3, padding='same', activation=LeakyReLU(0.3), kernel_initializer=initializer)
        self.In2 = tfa.layers.InstanceNormalization()

    def call(self, inputs):
        #block A
        x = self.conv1(inputs)
        output1 = self.In1(x)
        if self.down:
            output1 = self.AVP1(output1)

        #block B
        if self.down:
            output2 = self.AVP2(inputs)
        output2 = self.conv2(output2)
        output2 = self.In2(output2)
        return output1 + output2


def encoder():
    inputs = Input((64, 64, 1))
    out1 = res_block(64)(inputs)
    out2 = res_block(64)(out1)
    out3 = res_block(64)(out2)
    out4 = res_block(1)(out3)
    out5 = out4 + inputs
    out = Flatten()(out5)
    out = Dense(512, activation=LeakyReLU(0.4))(out)
    out = Dropout(0.4)(out)
    out = Dense(200, activation='tanh')(out)
    encoder = Model(inputs, [out, out4, out3, out2, out1])
    encoder.summary()
    return encoder

def decoder():
    #baseline result decoder input must be 256 dimension. normal has 200 dimension
    input = Input((200))
    d1 = Dense(4 * 4 * 512, use_bias=False, name="d1")(input)
    ac1 = LeakyReLU(0.3, name="ac1")(d1)
    reshape = Reshape((4, 4, 512), name="reshape")(ac1)

    up1 = residual_block_up(512)(reshape)
    up2 = residual_block_up(256)(up1)
    up3 = residual_block_up(128)(up2)
    up4 = residual_block_up(64)(up3)

    output = res_block(16)(up4)
    output = Conv2D(1, 3, activation="sigmoid", padding="same")(output)

    model = Model(input, output)
    model.summary()
    return model

def generator():
    input = Input((200))
    d1 = Dense(4 * 4 * 512, use_bias=False, name="d1")(input)
    ac1 = LeakyReLU(0.3, name="ac1")(d1)
    reshape = Reshape((4, 4, 512), name="reshape")(ac1)

    up1 = residual_block_up(512)(reshape)
    up2 = residual_block_up(256)(up1)
    up3 = residual_block_up(128)(up2)
    up4 = residual_block_up(64)(up3)

    output = res_block(16)(up4)
    output = Conv2D(1, 3, activation="sigmoid", padding="same")(output)

    model = Model(input, output)
    model.summary()
    return model

def discriminator():
    input1 = Input((64, 64, 1))
    out = Conv2D(16, 4, strides=(2, 2), padding="same")(input1)
    out = BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(0.3)(out)

    out = Conv2D(32, 4, strides=(2, 2), padding="same")(out)
    out = BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(0.3)(out)

    out = Conv2D(64, 4, strides=(2, 2), padding="same")(out)
    out = BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(0.3)(out)

    out = Conv2D(256, 4, strides=(2, 2), padding="same")(out)
    out = BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(0.3)(out)

    out = Conv2D(256, 4, padding="same")(out)
    out = BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(0.3)(out)

    out = Conv2D(1, 4, activation='sigmoid', padding="same")(out)
    model = Model(input1, out)
    model.summary()

    return model

def regression():
    def residual_block(input_data, dimentional):
        out = tf.keras.layers.Dense(dimentional, activation=LeakyReLU(0.3))(input_data)
        # out = tfa.layers.InstanceNormalization()(out)
        out = Dropout(0.4)(out)
        out = tf.keras.layers.Dense(dimentional, activation=LeakyReLU(0.3))(out)
        out = tf.keras.layers.add([input_data, out])
        return out

    input_data = Input((200))
    out = tf.keras.layers.Dense(200, activation=LeakyReLU(0.3))(input_data)
    x = out
    for i in range(6):
        x = residual_block(x, 200)
        if i == 1:
            output1 = x
        if i == 3:
            output2 = x
    output3= Dense(200, activation='tanh')(x)
    model = Model(input_data, [output1, output2, output3])
    model.summary()
    return model


def classifier():
    # 定義模型的輸入層
    inputs = tf.keras.Input((64, 64, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(111, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def cls():
    # 定義模型的輸入層
    inputs = tf.keras.Input((200))
    out = Dense(100, activation='relu')(inputs)
    out = Dropout(0.3)(out)
    out = BatchNormalization()(out)
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.3)(out)
    feature = BatchNormalization()(out)
    out = Dense(90, activation='softmax')(feature)
    # model = tf.keras.Model(inputs=inputs, outputs=[feature, out])
    model = tf.keras.Model(inputs=inputs, outputs = out)
    model.summary()
    return model

def cls_hat():
    # 定義模型的輸入層
    inputs = tf.keras.Input((90))
    out = Dense(100, activation='relu')(inputs)
    out = Dropout(0.3)(out)
    out = BatchNormalization()(out)
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.3)(out)
    feature = BatchNormalization()(out)
    out = Dense(200, activation='tanh')(feature)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.summary()
    return model



def unet():
    inputs = Input((64, 64, 1))
    out1 = res_block(64)(inputs)
    out2 = res_block(64)(out1)
    out3 = res_block(64)(out2)
    out4 = res_block(1)(out3)
    out5 = out4 + inputs
    out6 = Flatten()(out5)
    out7 = Dense(512, activation=LeakyReLU(0.4))(out6)
    out8 = Dropout(0.4)(out7)

    out9 = Dense(200, activation='tanh')(out8)

    out10 = Dropout(0.4)(out9)
    out11 = Dense(512, activation=LeakyReLU(0.4))(out10)
    out12 = Dense(64 * 64, use_bias=False, name="d1")(out11)
    out13 = LeakyReLU(0.3, name="ac1")(out12)
    out14 = Reshape((64, 64, 1), name="reshape")(out13)
    out15 = tf.concat([out14, out4], axis=-1)
    out16 = res_block(64)(out15)
    out17 = tf.concat([out16, out3], axis=-1)
    out18 = res_block(64)(out17)
    out19 = tf.concat([out18, out2], axis=-1)
    out20 = res_block(64)(out19)
    out21 = tf.concat([out20, out1], axis=-1)
    out22 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(out21)
    out23 = Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding='same')(out22)

    encoder = Model(inputs, out23)
    encoder.summary()
    plot_model(encoder, to_file='model.png')
    return encoder



def re_generator():
    input1 = Input((200))
    input2 = Input((64, 64, 1))
    input3 = Input((64, 64, 64))
    input4 = Input((64, 64, 64))
    input5 = Input((64, 64, 64))


    out10 = Dropout(0.4)(input1)
    out11 = Dense(512, activation=LeakyReLU(0.4))(out10)
    out12 = Dense(64 * 64, use_bias=False, name="d1")(out11)
    out13 = LeakyReLU(0.3, name="ac1")(out12)
    out14 = Reshape((64, 64, 1), name="reshape")(out13)
    out15 = tf.concat([out14, input2], axis=-1)
    out16 = res_block(64)(out15)
    out17 = tf.concat([out16, input3], axis=-1)
    out18 = res_block(64)(out17)
    out19 = tf.concat([out18, input4], axis=-1)
    out20 = res_block(64)(out19)
    out21 = tf.concat([out20, input5], axis=-1)
    out22 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(out21)
    out23 = Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding='same')(out22)

    encoder = Model([input1, input2, input3, input4, input5], out23)
    encoder.summary()
    return encoder





