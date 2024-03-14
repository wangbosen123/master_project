import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE



dataset, metadata = tfds.load('cycle_gan/horse2zebra',with_info=True, as_supervised=True)
train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

buffer_size = 256
batch_size = 1

# Define the standard image size.
orig_img_size = (286, 286)
# Size of the random crops to be used during training.
input_img_size = (256, 256, 3)
# Weights initializer for the layers.
kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=[*input_img_size])
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img

train_horses = (train_horses.map(preprocess_train_image, num_parallel_calls=autotune).cache().shuffle(buffer_size).batch(batch_size))
train_zebras = (train_zebras.map(preprocess_train_image, num_parallel_calls=autotune).cache().shuffle(buffer_size).batch(batch_size))
test_horses = (test_horses.map(preprocess_test_image, num_parallel_calls=autotune).cache().shuffle(buffer_size).batch(batch_size))
test_zebras = (test_zebras.map(preprocess_test_image, num_parallel_calls=autotune).cache().shuffle(buffer_size).batch(batch_size))


_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_horses.take(4), train_zebras.take(4))):
    horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax[i, 0].imshow(horse)
    ax[i, 1].imshow(zebra)
plt.savefig("result_image/original_datasets")
plt.close()