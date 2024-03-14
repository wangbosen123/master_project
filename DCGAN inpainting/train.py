import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from load_data import *
from build_model import *
from loss import *
from pretrain import *


def train_step(code, target, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(code)
        gen_img = generator(code)
        fake = discriminator(gen_img)
        context_loss = image_loss(target, gen_img)/10
        prior_loss = generator_loss(fake)*30
        code_loss = context_loss + prior_loss
    gradient_code = tape.gradient(code_loss, code)
    code = code - learning_rate * gradient_code
    return context_loss, prior_loss, code

def train(epochs, code, target, learning_rate):
    context_loss_total = []
    prior_loss_total = []
    for epoch in range(1,epochs+1):
        if epoch == 200:
            learning_rate /= 10
        if epoch == 1000:
            learning_rate /= 10
        if epoch == 2000:
            learning_rate /= 10

        start = time.time()
        context_loss, prior_loss, code = train_step(code,target,learning_rate)
        context_loss_total.append(context_loss)
        prior_loss_total.append(prior_loss)
        print("_________________________________________________")
        print(f"the epoch is {epoch}")
        print(f"the context_loss is {context_loss_total[-1]}")
        print(f"the prior_loss is {prior_loss_total[-1]}")
        print(f"the output code is {code[0][0:10]}")
        print(f"the time is {time.time() - start} second")
        print(f"the learning rate is {learning_rate}")
        if epoch % 100 == 0 or epoch == 1:
            draw_samples(epoch, target, code)
    return context_loss_total, prior_loss_total, code


def draw_samples(epoch, target, code, path="/home/dodo/Downloads/DCGAN_inpainting/"):

    gen_img = generator(code)
    gen_img, target = tf.reshape(gen_img,[64,64]), tf.reshape(target, [64,64])

    ax, fig = plt.subplots(figsize=(15, 4))

    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.imshow(target, cmap="gray")
    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.imshow(gen_img, cmap="gray")
    plt.savefig(path + f"produce_image_{epoch} epochs")
    plt.close()







if __name__ == "__main__":
    #encoder = encoder()
    generator =  create_generator()
    discriminator = create_discriminator()
    
    #set checkpoint
    #checkpoint_encoder = tf.train.Checkpoint(encoder)
    #checkpoint_generator = tf.train.Checkpoint(generator)
    #checkpoint_discriminator = tf.train.Checkpoint(discriminator)
    # checkpoint_encoder.restore("model_weight/encoder/encoder_900")
    #checkpoint_generator.load_weights("/home/bosen/Desktop/DCGAN/model_weight/generator_500_weights")
    #checkpoint_discriminator.load_weights("/home/bosen/Desktop/DCGAN/model_weight/discriminator_500_weights")
    generator.load_weights("/home/bosen/Desktop/DCGAN/model_weight/generator_500_weights")
    discriminator.load_weights("/home/bosen/Desktop/DCGAN/model_weight/discriminator_500_weights")

    # train_path = load_path(train=False)
    # target = load_image(get_batch_data(train_path, 0, 1), inpainting=True)
    '''
    target = cv2.imread("occ_image.jpg", cv2.IMREAD_GRAYSCALE)
    target = target / 127.5 - 1
    target = target.reshape(1,64,64,1)
    '''
    #code = encoder(target)

    target = np.load("/disk2/DCGAN_yu/CK1/imag_dis/train_g/imag.npy")
    target = target[4]*2-1
    target = target.reshape(1,64,64,1)
    
    code = tf.random.normal([1, 100])


    context_loss, prior_loss, final_code = train(3000, code, target, 1e-2)

    plt.plot(context_loss)
    plt.title("the context loss ")
    plt.savefig(f"/home/dodo/Downloads/DCGAN_inpainting/context_loss.jpg")
    plt.close()

    plt.plot(prior_loss)
    plt.title("the prior loss ")
    plt.savefig(f"/home/dodo/Downloads/DCGAN_inpainting/prior_loss.jpg")
    plt.close()