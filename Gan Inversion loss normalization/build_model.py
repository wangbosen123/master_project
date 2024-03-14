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
    # model.summary()
    # plot_model(model, to_file='result/G_structure.png', show_shapes=True, dpi=64)
    return model

def discriminator():
    input_x_layer = Input((64, 64, 1))

    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(input_x_layer)
    # x = LayerNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, (4, 4), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = Flatten()(x)
    output = Dense(1)(x)

    model = Model(inputs=input_x_layer, outputs=output)
    return model

def Patch_discriminator():
    input_x_layer = Input((64, 64, 1))

    x = Conv2D(64, (4, 4), strides=(2, 2), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(input_x_layer)
    x = Conv2D(64, (4, 4), strides=(1, 1), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Conv2D(128, (4, 4), strides=(2, 2), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Conv2D(256, (4, 4), strides=(2, 2), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Conv2D(512, (4, 4), strides=(2, 2), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = Conv2D(512, (4, 4), strides=(1, 1), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tfa.layers.InstanceNormalization()(x)
    out = Conv2D(1, (4, 4), strides=(1, 1), activation=LeakyReLU(0.3), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)

    model = Model(inputs=input_x_layer, outputs=out)
    # model.summary()
    return model

def encoder():
    inputs = Input((64, 64, 1))
    out = res_block(64)(inputs)
    out = res_block(64)(out)
    out = res_block(64)(out)
    out = res_block(1)(out)
    out = out + inputs
    out = Flatten()(out)
    out = Dense(512, activation=LeakyReLU(0.4))(out)
    out = Dropout(0.4)(out)
    out = Dense(256, activation=LeakyReLU(0.4))(out)

    mean_layer = Dense(256, activation='tanh')(out)
    var_layer = Dense(256, activation='tanh')(out)

    encoder = Model(inputs, [mean_layer, var_layer])
    # encoder.summary()
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
    cls = Dense(90, activation='softmax')(out)
    model = Model(inputs, [out, cls])
    # model.summary()
    return model

def ZtoZg():
    inputs = Input(256)
    feature = Dense(256, activation=LeakyReLU(0.3))(inputs)
    out = Dropout(0.3)(feature)
    cls = Dense(90, activation='softmax')(out)
    out = feature + out
    res = Dense(200, activation=LeakyReLU(0.3))(out)
    out = Dropout(0.3)(res)
    out = res + out
    out = Dense(200, activation='tanh')(out)
    model = Model(inputs, [out, feature, cls])
    # model.summary()
    return model

def H_cls():
    inputs = Input((3, 200))
    out = Conv1D(64, 1, padding='same', activation=LeakyReLU(0.3), strides=1)(inputs)
    out = Dropout(0.3)(out)
    # out = Conv1D(64, 1, padding='same', activation=LeakyReLU(0.3), strides=1)(out)
    # out = Dropout(0.3)(out)
    # out = Conv1D(64, 1, padding='same', activation=LeakyReLU(0.3), strides=1)(out)
    # out = Dropout(0.3)(out)
    feature = Flatten()(out)
    out = Dense(111, activation='softmax')(feature)
    model = Model(inputs, [feature, out])
    # model.summary()
    return model

def H_cls_no_condition():
    inputs = Input((200))
    out = Dense(200, activation=LeakyReLU(0.3))(inputs)
    feature = Dropout(0.3)(out)
    out = Dense(111, activation='softmax')(feature)
    model = Model(inputs, [feature, out])
    # model.summary()
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

if __name__ == '__main__':
    cls = H_cls()