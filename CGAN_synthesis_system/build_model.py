import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model


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


def normal_encoder():
    inputs = Input((64, 64, 1))
    out = res_block(64)(inputs)
    out = res_block(64)(out)
    out = res_block(64)(out)
    out = res_block(1)(out)
    out = out + inputs
    out = Flatten()(out)
    out = Dense(512, activation=LeakyReLU(0.4))(out)
    out = Dropout(0.4)(out)
    out = Dense(256, activation='tanh')(out)

    encoder = Model(inputs, out)
    # encoder.summary()
    # plot_model(encoder, to_file='result/encoder_structure.png', show_shapes=True, dpi=64)
    return encoder

def ZtoZd():
    inputs = Input(256)
    res = Dense(256, activation=LeakyReLU(0.3))(inputs)
    out = Dropout(0.3)(res)
    out = res + out
    res = Dense(200, activation=LeakyReLU(0.3))(out)
    out = Dropout(0.3)(res)
    out = res + out
    out = Dense(200, activation='tanh')(out)
    cls = Dense(91, activation='softmax')(out)
    model = Model(inputs, [out, cls])
    model.summary()
    return model

def ZtoZg():
    inputs = Input(256)
    feature = Dense(256, activation=LeakyReLU(0.3))(inputs)
    out = Dropout(0.3)(feature)
    cls = Dense(91, activation='softmax')(out)
    out = feature + out
    res = Dense(200, activation=LeakyReLU(0.3))(out)
    out = Dropout(0.3)(res)
    out = res + out
    out = Dense(200, activation='tanh')(out)
    model = Model(inputs, [out, feature, cls])
    model.summary()
    return model

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
    # model.summary()
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
    # plot_model(model, to_file='result/G_structure.png', show_shapes=True, dpi=64)
    return model

def patch_discriminator():
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

def regression_model_with_instance():
    def residual_block(input_data, dimentional):
        out = tf.keras.layers.Dense(dimentional, activation=LeakyReLU(0.3))(input_data)
        out = tfa.layers.InstanceNormalization()(out)
        out = tf.keras.layers.Dense(dimentional, activation=LeakyReLU(0.3))(out)
        out = tf.keras.layers.add([input_data, out])
        return out

    input_data = Input((200))
    cond = Input((200))
    input = Concatenate()([input_data, cond])
    out = tf.keras.layers.Dense(200, activation=LeakyReLU(0.3))(input)
    x = out
    for _ in range(6):
        x = residual_block(x, 200)
    output = Dense(200, activation='tanh')(x)
    model = Model([input_data, cond], output)
    model.summary()
    return model

# def regression_model_with_instance():
#
#     input_data = tf.keras.layers.Input((200))
#     cond = tf.keras.layers.Input((200))
#     input = tf.keras.layers.Concatenate()([input_data, cond])
#     out = tf.keras.layers.Dense(200, activation='relu')(input)
#     out = tfa.layers.InstanceNormalization()(out)
#     out = tf.keras.layers.Dense(200, activation='relu')(out)
#     res1 = tf.keras.layers.add([input_data, out])
#
#     start2 = tf.keras.layers.Dense(200, activation='relu')(res1)
#     out = tfa.layers.InstanceNormalization()(start2)
#     out = tf.keras.layers.Dense(200, activation='relu')(out)
#     res2 = tf.keras.layers.add([input_data, res1, out, start2])
#
#     start3 = tf.keras.layers.Dense(200, activation='relu')(res2)
#     out = tfa.layers.InstanceNormalization()(start3)
#     out = tf.keras.layers.Dense(200, activation='relu')(out)
#     res3 = tf.keras.layers.add([input_data, res1, res2, out, start3])
#
#     start4 = tf.keras.layers.Dense(200, activation='relu')(res3)
#     out = tfa.layers.InstanceNormalization()(start4)
#     out = tf.keras.layers.Dense(200, activation='relu')(out)
#     res4 = tf.keras.layers.add([input_data, res1, res2, res3, out, start4])
#
#     start5 = tf.keras.layers.Dense(200, activation='relu')(res4)
#     out = tf.keras.layers.Dense(200, activation='relu')(start5)
#     res5 = tf.keras.layers.add([input_data, res1, res2, res3, res4, out, start5])
#
#     output = Dense(200, activation='tanh')(res5)
#     model = Model([input_data, cond], output)
#     model.summary()
#     return model


def regression_model_without_instance():
    def residual_block(input_data, dimentional):
        out = tf.keras.layers.Dense(dimentional, activation=LeakyReLU(0.3))(input_data)
        # out = tfa.layers.InstanceNormalization()(out)
        out = tf.keras.layers.Dense(dimentional, activation=LeakyReLU(0.3))(out)
        out = tf.keras.layers.add([input_data, out])
        return out

    input_data = Input((200))
    x = input_data
    for _ in range(6):
        x = residual_block(x, 200)
    output = Dense(200, activation='tanh')(x)
    model = Model(input_data, output)
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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
regression_model_with_instance()




