import tensorflow as tf
from tensorflow.keras.losses import *
import numpy as np
import cv2
from build_model import *

def Local_discriminator_loss(real_output, fake_output):
    gt_real = tf.concat([tf.ones((real_output.shape[0], real_output.shape[1], real_output.shape[2], 1), dtype='float32'),
                   tf.zeros((real_output.shape[0], real_output.shape[1], real_output.shape[2], 1), dtype='float32')], axis=-1)

    gt_fake = tf.concat([tf.zeros((fake_output.shape[0], fake_output.shape[1], fake_output.shape[2], 1), dtype='float32'),
                   tf.ones((fake_output.shape[0], fake_output.shape[1], fake_output.shape[2], 1), dtype='float32')], axis=-1)

    return tf.reduce_mean(tf.square(gt_real-real_output)) + tf.reduce_mean(tf.square(gt_fake-fake_output))

# BCE loss
# def discriminator_loss(real_output, fake_output):
#     cross_entropy = tf.keras.losses.BinaryCrossentropy()
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss

# BCE loss
# def generator_loss(fake_output):
#     cross_entropy = tf.keras.losses.BinaryCrossentropy()
#     return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss(fake_output):
    return tf.reduce_mean(tf.square(fake_output - 1))



def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(tf.square(real_output - 1) + tf.square(fake_output))



#shape must be (-1,128,128,1)
def perceptual_loss(real_high, fake_high):
    real_high, fake_high = tf.cast(real_high,dtype="float32"), tf.cast(fake_high, dtype="float32")
    real_high = tf.image.grayscale_to_rgb(real_high)
    fake_high = tf.image.grayscale_to_rgb(fake_high)

    feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")


    real_feature = feature_extraction(real_high)
    fake_feature = feature_extraction(fake_high)
    distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
    return distance


if __name__ == "__main__":
    pass

