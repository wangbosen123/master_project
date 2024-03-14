import tensorflow as tf
from tensorflow.keras.losses import *
from build_model import *


def cycle_consistency_loss(real_images, generated_images):
    return tf.reduce_mean(tf.abs(real_images - generated_images))

def identity_loss(real_images, same_images):
    return tf.reduce_mean(tf.abs(real_images - same_images))


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)


if __name__ == "__main__":
    pass



