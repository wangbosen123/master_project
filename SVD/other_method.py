import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *

def model():
    inputs = Input((28,28,1))
    res = Conv2D(64,3,activation="relu",padding="same")(inputs)
    out = BatchNormalization()(res)
    out = Conv2D(64,3,activation="relu",padding="same")(out)
    res = Conv2D(64, 3, activation="relu", padding="same")(out)
    out = BatchNormalization()(res)
    out = Conv2D(64, 3, activation="relu", padding="same")(out)
    res = Conv2D(64, 3, activation="relu", padding="same")(out)
    out = BatchNormalization()(res)
    out = Conv2D(64, 3, activation="relu", padding="same")(out)
    flat = Flatten()(out)
    out = Dense(128,activation="relu")(flat)
    out = Dense(10, activation="softmax")(out)
    model = Model(inputs, out)
    model.summary()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = model()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


X_train = X_train.reshape(-1, 28,28,1)/255
X_test = X_test.reshape(-1, 28,28,1)/255.


y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)
model.fit(X_train, y_train, epochs=8, batch_size=60,verbose=1,validation_data=(X_test,y_test))
