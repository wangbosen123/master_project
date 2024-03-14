from prepare_data import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def geometry_predictor():
    input = Input((64, 64, 3))
    out = Conv2D(4, 7, strides=(1, 1), padding='valid', activation='relu')(input)
    out = Conv2D(8, 3, strides=(1, 1), padding='valid', activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Flatten()(out)
    out = Dense(48, activation='relu')(out)
    out = Dense(6, )(out)
    model = Model(input, out)
    model.summary()
    return model

def cls():
    input = Input((64, 64, 1))
    x = Conv2D(64, 4, padding='same', strides=(1, 1), activation="relu")(input)
    x = Conv2D(64, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = Conv2D(128, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(256, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = Conv2D(256, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(512, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = Conv2D(512, 4, padding='same', strides=(1, 1), activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(90, activation='softmax')(x)
    model = Model(input, output)
    model.summary()
    return model

if __name__ == '__main__':
    pass
