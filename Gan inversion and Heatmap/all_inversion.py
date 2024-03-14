import cv2

import highfaceGAN
import loss
from build_model import *
from highfaceGAN import *
from prepare_data import *
from loss import *


class inversionGAN:
    def __init__(self, epochs, learning_rate):
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

    def train_step(self, real_high, code):
        real_high = real_high.reshape(-1,64,64,1)
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(code)
            fake_high = self.generator(code)
            fake = self.discriminator(fake_high)
            pr_loss_real_fake = generator_loss(fake)

            pr_loss = pr_loss_real_fake
            perceptual_loss = 100*loss.perceptual_loss(real_high, fake_high)
            rec_loss = 1000 * tf.reduce_mean(tf.square(real_high - fake_high))
            total_loss = rec_loss + perceptual_loss + pr_loss
        gradient_code = code_tape.gradient(total_loss,code)
        code = code - self.learning_rate * gradient_code

        return rec_loss, perceptual_loss, pr_loss , code


    def train(self):
        #using the test data celeba >> after 40000
        train_path, test_path = load_path()
        for number in range(20):
            real_low, real_high = load_image(get_batch_data(train_path, number,300), train=True)
            real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)
            code = self.encoder(real_low)
            rec_loss_epoch = []
            perceptual_loss_epoch = []
            prior_loss_epoch = []
            print("the next batch__________________________________________________________")
            for epoch in range(self.epochs):
                start = time.time()
                rec_loss, perceptual_loss, pr_loss, code = self.train_step(real_high, code)
                rec_loss_epoch.append(rec_loss)
                perceptual_loss_epoch.append(perceptual_loss)
                prior_loss_epoch.append(pr_loss)

                print("______________________________________")
                print(f"the epoch is {epoch+1}")
                print(f"the mse_loss is {rec_loss_epoch[-1]}")
                print(f"the perceptual_loss is {perceptual_loss_epoch[-1]}")
                print(f"the pr_loss is {prior_loss_epoch[-1]}")
                print(f"the new_code is {code[0][0:10]}")
                print("the spend time is %s second" % (time.time() - start))
                # print(f"the learning_rate is {self.learning_rate}")
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    self.draw_sample(number, epoch+1, code)

            plt.plot(rec_loss_epoch)
            plt.title("the rec_loss")
            plt.savefig(f"result/total_inversion_result_image/train_rec_loss{number}.jpg")
            plt.close()

            plt.plot(perceptual_loss_epoch)
            plt.title("the perceptual_loss")
            plt.savefig(f"result/total_inversion_result_image/train_perceptual_loss{number}.jpg")
            plt.close()

            plt.plot(prior_loss_epoch)
            plt.title(f"the prior_loss")
            plt.savefig(f"result/total_inversion_result_image/train_prior_loss{number}.jpg")
            plt.close()
            np.savetxt(f"result/inverted_code/train_{number}_batch.csv", code, delimiter=",")


    def draw_sample(self, number, epoch, code, path = "result/total_inversion_result_image"):

        #using the test data part of celeba dataset
        train_path, test_path = load_path()
        real_low, real_high = load_image(get_batch_data(train_path, number, 400), train=True)
        real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)
        code = code[0:5]

        original_code = self.encoder(real_low)
        original_fake_high = self.generator(original_code)
        original_fake_high = tf.reshape(original_fake_high, [-1, 64, 64])

        update_fake_high = self.generator(code)
        updata_fake_high = tf.reshape(update_fake_high, [-1,64,64])



        ax,fig = plt.subplots(figsize=(6,5))
        for i in range(5):
            plt.subplot(4, 5, i+1)
            plt.axis("off")
            plt.imshow(real_high[i],cmap="gray")
            plt.subplot(4, 5, i+6)
            plt.axis("off")
            plt.imshow(real_low[i],cmap="gray")
            plt.subplot(4, 5, i + 11)
            plt.axis("off")
            plt.imshow(original_fake_high[i], cmap="gray")
            plt.subplot(4, 5, i + 16)
            plt.axis("off")
            plt.imshow(updata_fake_high[i], cmap="gray")

        plt.savefig(path + "/" + f"train_{number}_batch_{epoch}.jpg")
        plt.close()



if __name__ == "__main__":
    inversion = inversionGAN(3000, 3000)
    code = inversion.train()