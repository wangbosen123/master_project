from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D
import tensorflow as tf
import tensorflow_addons as tfa

class IC_STN(tf.keras.Model):

    def __init__(self, warpDim):    # warpDim = opt.warpDim
        super(IC_STN, self).__init__()
        # GP
        self.conv1 = Conv2D(4,
                            kernel_size=(7, 7),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv1',
                            trainable=True)
        self.conv2 = Conv2D(8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv2',
                            trainable=True)
        self.pool = MaxPool2D(pool_size=(2, 2),
                              strides=(1, 1),
                              padding='valid',
                              name='pool')
        self.flat = Flatten()

        self.fc0 = Dense(48, activation='relu', name='fc0', trainable=True)

        self.fc1 = Dense(warpDim, activation='tanh', trainable=True)

    def call(self, inputs, training=True, **kwargs):
        z = self.conv1(inputs)
        z = self.conv2(z)
        z = self.pool(z)
        z = self.flat(z)
        z = self.fc0(z)
        z = self.fc1(z)
        return z

# --

class residual_block_down(tf.keras.Model):

    def __init__(self, channels, down=True):
        super(residual_block_down, self).__init__()
        initializer = tf.keras.initializers.Orthogonal()
        self.down = down
        self.conv1 = Conv2D(channels, 1, activation="relu",  kernel_initializer=initializer)
        self.bn1 = BatchNormalization()
        if down:
            self.AVP1 = AveragePooling2D()
            self.AVP2 = AveragePooling2D()
        self.conv2 = Conv2D(channels, 3, padding='same', activation="relu", kernel_initializer=initializer)
        self.conv3 = Conv2D(channels, 3, padding='same', activation="relu", kernel_initializer=initializer)
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def call(self, inputs):
        #block A
        x = self.conv1(inputs)
        output1 = self.bn1(x)
        if self.down:
            output1 = self.AVP1(output1)
        #block B
        if self.down:
            output2 = self.AVP2(inputs)
        output2 = self.conv2(output2)
        output2 = self.bn2(output2)
        output2 = self.conv3(output2)
        output2 = self.bn3(output2)
        return output1 + output2

class Enc(tf.keras.Model):

    def __init__(self):
        super(Enc, self).__init__()
        self.enc_res1 = residual_block_down(64)
        self.enc_res2 = residual_block_down(128)
        self.enc_res3 = residual_block_down(256)
        self.enc_res4 = residual_block_down(512)
        self.enc_res5 = residual_block_down(1024)

    def call(self, inputs, **kwargs):
        z = self.enc_res1.call(inputs)  # (1, 64, 64, 64)
        z = self.enc_res2.call(z)  # (1, 32, 32, 128)
        z = self.enc_res3.call(z)  # (1, 16, 16, 256)
        z = self.enc_res4.call(z)  # (1, 8, 8, 512)
        z1 = self.enc_res5.call(z)  # (1, 4, 4, 1024)
        return z1

class ID_Cls(tf.keras.Model):

    def __init__(self):
        super(ID_Cls, self).__init__()

        self.flatten = Flatten()

        self.fc1 = Dense(1024, name='fc2', activation='sigmoid', trainable=True)

        self.fc2 = Dense(256, name='fc3', activation='sigmoid', trainable=True)

        self.fc3 = Dense(90, name='fc4', trainable=True)

    def call(self, A):
        z = self.flatten(A)       # (1, 8*8*128)  8192
        z = self.fc1(z)              # (1, 1024)
        z = self.fc2(z)              # (1, 256)
        pred = self.fc3(z)           # (1, 90)
        return pred
