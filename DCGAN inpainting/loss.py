import tensorflow as tf
from tensorflow.keras.losses import *
import numpy as np
from load_data import *

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def image_loss(real,pred, mask=True):
    real = tf.reshape(real,[64,64])
    pred = tf.reshape(pred,[64,64])
    mae = tf.keras.losses.MeanSquaredError()
    if mask:
        m = make_mask()
        loss = tf.reduce_sum(abs(m*(real - pred)))
        return loss
    else:
        return mae(real, pred)




