import cv2
import numpy as np
from load_data import *
from build_model import *
from loss import *
import tensorflow as tf
import matplotlib.pyplot as plt


generator = generator()
encoder = encoder()

checkpoint_encoder = tf.train.Checkpoint(encoder)
checkpoint_generator = tf.train.Checkpoint(generator)
checkpoint_encoder.restore("model_weight/encoder/encoder-900")
checkpoint_generator.restore("model_weight/generator/generator-140")


image = cv2.imread("occ_image.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.rectangle(image,(22,22),(43,43),(0,0,0),-1)
# cv2.imwrite("occ_image.jpg", image)
image = image / 127.5 -1
image = image.reshape(1,64,64,1)
code = encoder(image)
# print(code)
# code = tf.random.uniform([1,200],maxval=10,minval=-10)
# # code = tf.random.normal([1,200],stddev=1,mean=0)
gen_img = generator(code)
gen_img = tf.reshape(gen_img, [64,64])
gen_img = (np.array(gen_img)+1)*127.5
# # print(gen_img)
cv2.imwrite("produce_occ_image.jpg", gen_img)
# plt.imshow(gen_img, cmap="gray")
# plt.savefig("produce_occ_image.jpg")



































# def test_pretrain():
#     generator = build_generator()
#     generator.load_weights("model_weight/generator_548_weights")
#     noise1 = tf.random.uniform(shape=[1,100],minval=-1,maxval=1,dtype="float32")
#     noise2 = tf.random.uniform(shape=[1,100],minval=-1,maxval=1,dtype="float32")
#     gen_img1 = generator(noise1)
#     gen_img2 = generator(noise2)
#     gen_img1 = tf.reshape(gen_img1,[64,64])
#     gen_img2 = tf.reshape(gen_img2,[64,64])
#     plt.imshow(gen_img2,cmap="gray")
#     plt.show()
#     plt.savefig("result_image/img2.jpg")
#     plt.imshow(gen_img1,cmap="gray")
#     plt.show()
#     plt.savefig("result_image/img1.jpg")
#
#
#
# if __name__ == "__main__":
#     test_pretrain()
#     # noise = tf.random.uniform(shape=[1,5],minval=-1,maxval=1,dtype="float32")
#     # print(noise)
#     # with tf.GradientTape() as code_tape:
#     #     code_tape.watch(noise)
#     #     loss = 1-noise
#     # gradient_code = code_tape.gradient(loss,noise)
#     #
#     # print(gradient_code)



