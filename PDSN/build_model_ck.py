import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *



class Res_block(Model):
    def __init__(self,output_plain,stride):
        super(Res_block, self).__init__()
        self.BN1 = BatchNormalization()
        self.conv1 = Conv2D(output_plain, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_normal')
        self.ac1 = LeakyReLU(0.4)
        self.BN2 = BatchNormalization()
        self.conv2 = Conv2D(output_plain, kernel_size=3, strides=stride, padding='same',kernel_initializer='glorot_normal')

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(Conv2D(output_plain, kernel_size=1, strides=stride, padding='same',kernel_initializer='glorot_normal'))
            self.downsample.add(BatchNormalization())

    def call(self, inputs, training=False, **kwargs):
        res = self.downsample(inputs)
        out = self.BN1(inputs)
        out = self.conv1(out)
        out = self.ac1(out)
        out = self.BN2(out)
        out = self.conv2(out)
        res += out
        return res


class PDSN_model(Model):
    def __init__(self):
        super(PDSN_model, self).__init__()
        ### Siamese Network
        self.layer0 = self.make_basic_block_layer(128, 2)
        self.layer1 = self.make_basic_block_layer(256, 2)
        self.layer2 = self.make_basic_block_layer(512, 2)
        self.layer3 = self.make_basic_block_layer(1024, 2)

        ### mask generator
        self.conv = Conv2D(1024, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal', activation=LeakyReLU(0.4))
        self.BN1 = BatchNormalization()

        ### Cls for Reg mnist
        self.flatten = Flatten()
        self.NN1 = Dense(512, kernel_initializer='glorot_normal', activation='sigmoid')
        self.dropout1 = Dropout(0.3)
        self.NN2 = Dense(128, kernel_initializer="glorot_normal", activation="sigmoid")
        self.dropout2 = Dropout(0.3)
        self.NN3 = Dense(2, kernel_initializer='glorot_normal', activation='softmax')


    def make_basic_block_layer(self, filter_num, stride):
        return Res_block(filter_num, stride)

    def Siamese_Network(self,x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    def mask_generator(self,x):
        out = self.conv(x)
        out = self.BN1(out)
        out = tf.math.sigmoid(out)
        return out

    def Cls(self, x):
        out = self.flatten(x)
        out = self.NN1(out)
        out = self.dropout1(out)
        out = self.NN2(out)
        out = self.dropout2(out)
        out = self.NN3(out)
        return out

    def call(self, inputs, training=False, **kwargs):
        Nonocc_image, Occ_image = inputs[:,:,:,0], inputs[:,:,:,1]
        Nonocc_image, Occ_image = tf.expand_dims(Nonocc_image,axis=-1), tf.expand_dims(Occ_image,axis=-1)
        Nonocc_feature, Occ_feature = self.Siamese_Network(Nonocc_image), self.Siamese_Network(Occ_image)
        Z_different = tf.abs(Nonocc_feature - Occ_feature)
        Z_different = self.mask_generator(Z_different)
        Nonocc_feature_x_Z_different = tf.multiply(Z_different,Nonocc_feature)
        Occ_feature_x_Z_different = tf.multiply(Z_different, Occ_feature)
        pred_label_for_Occ = self.Cls(Occ_feature_x_Z_different)
        pred_label_for_nocc = self.Cls(Nonocc_feature_x_Z_different)
        return Z_different, Nonocc_feature_x_Z_different,Occ_feature_x_Z_different,pred_label_for_nocc, pred_label_for_Occ


if __name__=="__main__":
    model = PDSN_model()
    model.build((1, 128, 128, 2))
    model.summary()

    for layer in model.layers:
        print(layer,layer.trainable)


