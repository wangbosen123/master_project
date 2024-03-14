import cv2

import highfaceGAN
import loss
from build_model import *
from highfaceGAN import *
from prepare_data import *
from loss import *


class inversionGAN:
    def __init__(self, target, ground_truth, epochs, learning_rate):
        self.target = target / 255
        self.ground_truth = ground_truth / 255
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.encoder = encoder()
        self.generator = generator()
        self.discriminator = discriminator()

        self.checkpoint_encoder = tf.train.Checkpoint(self.encoder)
        self.checkpoint_generator = tf.train.Checkpoint(self.generator)
        self.checkpoint_discriminator = tf.train.Checkpoint(self.discriminator)
        self.checkpoint_encoder.restore("model_weight/instance_encoder/encoder-300")
        self.checkpoint_generator.restore("model_weight/instance_generator/generator-300")
        self.checkpoint_discriminator.restore("model_weight/instance_discriminator/discriminator-300")

    def train_step(self,code):
        ground_truth = self.ground_truth.reshape(1,64,64,1)
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(code)
            gen_img = self.generator(code)
            fake = self.discriminator(gen_img)
            pr_loss = generator_loss(fake)


            perceptual_loss = 100*loss.perceptual_loss(ground_truth, gen_img)
            rec_loss = 1000 * tf.reduce_mean(tf.square(ground_truth - gen_img))

            total_loss = rec_loss + perceptual_loss + pr_loss
        gradient_code = code_tape.gradient(total_loss,code)
        code = code - self.learning_rate * gradient_code

        return rec_loss, perceptual_loss, pr_loss, code


    def train(self):
        rec_loss_total = []
        perceptual_loss_total = []
        prior_loss_total = []
        target = self.target.reshape(1, 64, 64, 1)
        code = self.encoder(target)
        gen_img = self.generator(code)
        gen_img = tf.reshape(gen_img, [64,64])
        print(1000 * tf.reduce_mean(tf.square(self.ground_truth - gen_img)))

        for epoch in range(self.epochs):
            start = time.time()
            rec_loss, perceptual_loss, pr_loss, code = self.train_step(code)
            rec_loss_total.append(rec_loss)
            perceptual_loss_total.append(perceptual_loss)
            prior_loss_total.append(pr_loss)
            end = time.time()
            print("______________________________________")
            print(f"the epoch is {epoch+1}")
            print(f"the mse_loss is {rec_loss_total[-1]}")
            print(f"the perceptual_loss is {perceptual_loss_total[-1]}")
            print(f"the pr_loss is {prior_loss_total[-1]}")
            print(f"the new_code is {code[0][0:10]}")
            print("the spend time is %s second" % (end-start))
            # print(f"the learning_rate is {self.learning_rate}")
            if (epoch + 1) % 100 == 0 or epoch == 0:
                self.draw_sample(epoch+1, code)

        return rec_loss_total, perceptual_loss_total, prior_loss_total, code

    def draw_sample(self, epoch, code, path="result/single_inversion_result_image"):
        target = self.target.reshape(1, 64, 64, 1)
        original_code = self.encoder(target)
        original_fake_img = self.generator(original_code)
        fake_high = self.generator(code)

        original_fake_img = tf.reshape(original_fake_img, [64,64])
        fake_high = tf.reshape(fake_high,[64,64])
        target = tf.reshape(self.target, [64,64])
        ax,fig = plt.subplots(figsize=(10,4))
        plt.subplot(2,2,1)
        plt.axis("off")
        plt.imshow(self.ground_truth, cmap="gray")
        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.imshow(original_fake_img,cmap="gray")
        plt.subplot(2,2,3)
        plt.axis("off")
        plt.imshow(target,cmap="gray")
        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.imshow(fake_high, cmap="gray")

        plt.savefig(path + "/" + f"{epoch}.jpg")
        plt.close()


        if epoch == 2901 or epoch == 2900:
            fake_high = np.array(fake_high)*255
            cv2.imwrite("inversion_domain_result/inversion_target.jpg", fake_high)






if __name__ == "__main__":
    #load_model_weights
    # encoder = encoder()
    # generator = generator()
    # checkpoint_encoder = tf.train.Checkpoint(encoder)
    # checkpoint_generator = tf.train.Checkpoint(generator)
    # checkpoint_encoder.restore("model_weight/instance_encoder/encoder-300")
    # checkpoint_generator.restore("model_weight/instance_generator/generator-300")

    # #deal the single image
    ground_truth = cv2.imread("celeba_test/15_test.jpg", cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(ground_truth, (5, 5), sigmaX=1, sigmaY=1)
    low_img = cv2.resize(blur, (8, 8), cv2.INTER_CUBIC)
    target = cv2.resize(low_img, (64, 64), cv2.INTER_CUBIC)

    # only encoder
    # target = target / 255
    # target = target.reshape(1,64,64,1)
    # init_code = encoder(target)
    # init_image = generator(init_code)
    # init_image = tf.reshape(init_image, [64,64]).numpy()*255
    # cv2.imwrite("result/single_inversion_result_image/encoder_init_image.jpg", init_image)



    # #check single image
    inversion = inversionGAN(target,ground_truth,3000,10)
    mse_loss, perceptual_loss, prior_loss, code= inversion.train()
    plt.plot(mse_loss)
    plt.title("the rec_loss")
    plt.savefig("result/single_inversion_result_image/rec_loss.jpg")
    plt.close()

    plt.plot(perceptual_loss)
    plt.title("the perceptual_loss")
    plt.savefig("result/single_inversion_result_image/perceptual_loss")
    plt.close()

    plt.plot(prior_loss)
    plt.title("the prior_loss")
    plt.savefig("result/single_inversion_result_image/prior_loss")
    plt.close()

