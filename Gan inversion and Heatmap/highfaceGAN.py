import loss
from prepare_data import *
from loss import *
import build_model
from prepare_data import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from loss import *
import time



def discriminator_train_step(real_low, real_high, d_optimizer):
    with tf.GradientTape() as tape:
        code = encoder(real_low)
        fake_high = generator(code)
        fake = discriminator(fake_high)
        real = discriminator(real_high)
        real_fake_loss = discriminator_loss(real, fake)
        dis_loss = 0.5 * real_fake_loss
    grads = tape.gradient(dis_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return dis_loss


def encoder_train_step(real_low, real_high, g_optimizer):
    with tf.GradientTape() as tape:
        code = encoder(real_low)
        fake_high = generator(code)
        fake = discriminator(fake_high)
        rec_loss = 70 * tf.reduce_mean(tf.square(fake_high - real_high))
        real_fake_loss = generator_loss(fake)
        adv_loss = real_fake_loss
        perceptual_loss = loss.perceptual_loss(real_high, fake_high)
        # perceptual_loss = 0
        encoder_loss = rec_loss + adv_loss + perceptual_loss
    grads = tape.gradient(encoder_loss, encoder.trainable_variables+generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, encoder.trainable_variables+generator.trainable_variables))
    return rec_loss, adv_loss, perceptual_loss

def training(epochs, batch_size, batch_num, beg=0):
    encoder_rec_loss_avg = []
    encoder_adv_loss_avg = []
    encoder_percetual_loss_avg = []
    dis_loss_avg = []
    g_optimizer = tf.keras.optimizers.Adam(1e-6)
    d_optimizer = tf.keras.optimizers.Adam(6e-6)
    # g_optimizer = tf.keras.optimizers.Adam(1e-7)
    # d_optimizer = tf.keras.optimizers.Adam(1e-7)
    for epoch in range(1, epochs):
        encoder_rec_loss_epoch = []
        encoder_adv_loss_epoch = []
        encoder_percetual_loss_epoch = []
        dis_loss_epoch = []
        start = time.time()
        train_path, _ = load_path()


        for batch in range(batch_num):
            print(batch, end=" ")
            real_low, real_high = load_image(get_batch_data(train_path, batch, batch_size), train=True)
            real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)
            for i in range(3):
                encoder_rec_loss, encoder_adv_loss, encoder_percetual_loss = encoder_train_step(real_low, real_high, g_optimizer)
                encoder_rec_loss_epoch.append(encoder_rec_loss)
                encoder_adv_loss_epoch.append(encoder_adv_loss)
                encoder_percetual_loss_epoch.append(encoder_percetual_loss)
        encoder_rec_loss_avg.append(np.mean(encoder_rec_loss_epoch))
        encoder_adv_loss_avg.append(np.mean(encoder_adv_loss_epoch))
        encoder_percetual_loss_avg.append(np.mean(encoder_percetual_loss_epoch))

        # choss one real_data, and gen_image to predict the D answer
        idx = np.random.randint(0, 1000)
        image = cv2.imread(f"celeba_train/{idx}_train.jpg", cv2.IMREAD_GRAYSCALE)
        low_image, _ = load_image(get_batch_data(train_path, idx, 1), train=True)
        code = encoder(low_image)
        fake_img = generator(code)
        image = image.reshape(1, 64, 64, 1) / 255

        print("_____________________________________________________")
        print(f"the epoch is {epoch+beg}_train_G")
        print(f"the encoder_reconstruction_loss is {encoder_rec_loss_avg[-1]}")
        print(f"the encoder_adv_loss is {encoder_adv_loss_avg[-1]}")
        print(f"the encoder_perceptual_loss is {encoder_percetual_loss_avg[-1]}")
        print("the spend time is : %s seconds " % (time.time() - start))
        print(f"G_Pred the fake image by D is {discriminator(fake_img)}")


        for batch in range(batch_num):
            print(batch, end=" ")
            real_low, real_high = load_image(get_batch_data(train_path, batch, batch_size), train=True)
            real_low, real_high =  real_low.reshape(-1, 64, 64, 1), real_high.reshape(-1,64,64,1)
            for i in range(10):
                dis_loss = discriminator_train_step(real_low, real_high, d_optimizer)
                dis_loss_epoch.append(dis_loss)
        dis_loss_avg.append(np.mean(dis_loss_epoch))
        print("____________________________________________________-")
        print(f"the epoch is {epoch+beg}__train_D")
        print(f"the dis_loss is {dis_loss_avg[-1]}")
        print("the spend time is : %s seconds " % (time.time() - start))
        print(f"D_Pred the real image by D is {discriminator(image)}")
        print(f"D_Pred the fake image by D is {discriminator(fake_img)}")



        if epoch % 10 == 0:
            manager_encoder.save(checkpoint_number=epoch+beg)
            manager_generator.save(checkpoint_number=epoch+beg)
            manager_discriminator.save(checkpoint_number=epoch+beg)



        draw_samples(epoch+beg, train=True)
        draw_samples(epoch+beg, train=False)
    return encoder_rec_loss_avg, encoder_adv_loss_avg, encoder_percetual_loss_avg, dis_loss_avg


def draw_samples(epoch,train=True,path="result/instance_highfacegan_result_image/"):
    train_path, test_path = load_path()
    if train:
        real_low, real_high = load_image(get_batch_data(train_path, 0, 20), train=True)
    else:
        real_low, real_high = load_image(get_batch_data(test_path, 0, 20), train=False)

    real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)

    code = encoder(real_low)
    fake_high = generator(code)
    fake_high = tf.reshape(fake_high, [-1, 64, 64])

    plt.subplots(figsize=(15, 4))
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.axis("off")
        plt.imshow(real_low[i], cmap="gray")
        plt.subplot(3, 10, i + 11)
        plt.axis("off")
        plt.imshow(fake_high[i], cmap="gray")
        plt.subplot(3, 10, i + 21)
        plt.axis("off")
        plt.imshow(real_high[i], cmap="gray")
    if train:
        plt.savefig(path + f"train_produce_image_{epoch}epochs")
    else:
        plt.savefig(path + f"test_produce_image_{epoch}epochs")

    plt.close()




if __name__ == "__main__":
    encoder = build_model.encoder()
    generator = build_model.generator()
    discriminator = build_model.discriminator()

    #set checkpoint
    checkpoint_encoder = tf.train.Checkpoint(encoder)
    checkpoint_generator = tf.train.Checkpoint(generator)
    checkpoint_discriminator = tf.train.Checkpoint(discriminator)
    checkpoint_encoder.restore("model_weight/instance_encoder/encoder-500")
    checkpoint_generator.restore("model_weight/instance_generator/generator-500")
    checkpoint_discriminator.restore("model_weight/instance_discriminator/discriminator-500")
    manager_encoder = tf.train.CheckpointManager(checkpoint_encoder, directory='model_weight/instance_encoder', max_to_keep=25, checkpoint_name="encoder")
    manager_generator = tf.train.CheckpointManager(checkpoint_generator, directory='model_weight/instance_generator', max_to_keep=25, checkpoint_name="generator")
    manager_discriminator = tf.train.CheckpointManager(checkpoint_discriminator, directory='model_weight/instance_discriminator', max_to_keep=25,checkpoint_name="discriminator")


    encoder_rec_loss, encoder_adv_loss, encoder_percetual_loss, dis_loss = training(2, 50, 120, 500)
    np.savetxt(f"result/instance_highfacegan_result_image/training_result.csv", encoder_rec_loss, delimiter=",")

    # plt.plot(encoder_rec_loss)
    # plt.title("the encoder_recontruction_loss")
    # plt.savefig("result/highfacegan_result_image/encoder_reconstruction_loss")
    # plt.close()
    #
    #
    # plt.plot(encoder_percetual_loss)
    # plt.title("the encoder_percetual_loss")
    # plt.savefig("result/highfacegan_result_image/encoder_percetual_loss")
    # plt.close()
    #
    # plt.plot(dis_loss)
    # plt.plot(encoder_adv_loss)
    # plt.title("the adv_loss")
    # plt.legend(["dis_adv","gen_adv"],loc="upper right")
    # plt.savefig("result/highfacegan_result_image/adv_loss")
    # plt.close()


