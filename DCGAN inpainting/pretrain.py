import loss
from load_data import *
from loss import *
import build_model
from load_data import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from loss import *
import time



def discriminator_train_step(train_data):
    with tf.GradientTape() as tape:
        code = encoder(train_data)
        fake = generator(code)
        fake = discriminator(fake)
        real = discriminator(train_data)
        real_fake_loss = discriminator_loss(real,fake)
        dis_loss = 0.5 * real_fake_loss
    grads = tape.gradient(dis_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return dis_loss


def encoder_train_step(train_data):
    with tf.GradientTape() as tape:
        code = encoder(train_data)
        fake_image = generator(code)
        fake = discriminator(fake_image)
        rec_loss = tf.reduce_mean(tf.square(fake_image - train_data))
        real_fake_loss = generator_loss(fake)

        # perceptual_loss = 10*loss.perceptual_loss(real_high, fake_high)
        encoder_loss = rec_loss + real_fake_loss
    grads = tape.gradient(encoder_loss, encoder.trainable_variables + generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + generator.trainable_variables))
    return rec_loss, real_fake_loss

def training(epochs,batch_size,batch_num):
    encoder_rec_loss_avg = []
    encoder_adv_loss_avg = []
    # encoder_percetual_loss_avg = []
    dis_loss_avg = []
    for epoch in range(1, epochs):
        encoder_rec_loss_epoch = []
        encoder_adv_loss_epoch = []
        # encoder_percetual_loss_epoch = []
        dis_loss_epoch = []
        start = time.time()
        for batch in range(batch_num):
            train_data = load_image(get_batch_data(load_path(train=True), batch, batch_size))
            test_data = load_image(get_batch_data(load_path(train=False), batch, batch_size))
            train_data, test_data = train_data.reshape(-1,64,64,1), test_data.reshape(-1, 64, 64, 1)
            dis_loss = discriminator_train_step(train_data)
            dis_loss_epoch.append(dis_loss)
            for i in range(2):
                encoder_rec_loss, encoder_adv_loss  = encoder_train_step(train_data)
                encoder_rec_loss_epoch.append(encoder_rec_loss)
                encoder_adv_loss_epoch.append(encoder_adv_loss)
                # encoder_percetual_loss_epoch.append(encoder_percetual_loss)


        encoder_rec_loss_avg.append(np.mean(encoder_rec_loss_epoch))
        encoder_adv_loss_avg.append(np.mean(encoder_adv_loss_epoch))
        # encoder_percetual_loss_avg.append(np.mean(encoder_percetual_loss_epoch))
        dis_loss_avg.append(np.mean(dis_loss_epoch))
        print(f"the epoch is {epoch+590}")
        print(f"the encoder_reconstruction_loss is {encoder_rec_loss_avg[-1]}")
        print(f"the encoder_adv_loss is {encoder_adv_loss_avg[-1]}")
        # print(f"the encoder_perceptual_loss is {encoder_percetual_loss_avg[-1]}")
        print(f"the dis_loss is {dis_loss_avg[-1]}")
        print("the spend time is : %s seconds " % (time.time() - start))

        if epoch % 10 == 0:
            manager_encoder.save(checkpoint_number=epoch+590)
            manager_generator.save(checkpoint_number=epoch+590)
            manager_discriminator.save(checkpoint_number=epoch+590)
        # if epoch > 100 and gen_loss_avg[-1] == min(gen_loss_avg[100:]):
        #     generator.save_weights(f"model_weight/generator_{epoch+454}_weights")
        #     discriminator.save_weights(f"model_weight/discriminator_{epoch+454}_weights")


        draw_samples(epoch+590, train=True)
        draw_samples(epoch+590, train=False)
    return encoder_rec_loss_avg, encoder_adv_loss_avg, dis_loss_avg

def draw_samples(epoch,train=True,path="result_image/"):
    if train:
        data = load_image(get_batch_data(load_path(train=True), 100, 10))
        data = data.reshape(-1,64,64,1)
    else:
        data = load_image(get_batch_data(load_path(train=False), 0, 20))
        data = data.reshape(-1,64,64,1)


    code = encoder(data)
    # for i in range(10):
    #     print(code[i][:20])

    fake = generator(code)
    fake= tf.reshape(fake, [-1, 64, 64])

    ax, fig = plt.subplots(figsize=(15, 4))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.axis("off")
        plt.imshow(data[i], cmap="gray")
        plt.subplot(2, 10, i + 11)
        plt.axis("off")
        plt.imshow(fake[i], cmap="gray")

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
    checkpoint_encoder.restore("model_weight/encoder/encoder-590")
    checkpoint_generator.restore("model_weight/generator/generator-590")
    checkpoint_discriminator.restore("model_weight/discriminator/discriminator-590")
    manager_encoder = tf.train.CheckpointManager(checkpoint_encoder, directory='model_weight/encoder', max_to_keep=10, checkpoint_name="encoder")
    manager_generator = tf.train.CheckpointManager(checkpoint_generator, directory='model_weight/generator', max_to_keep=30,checkpoint_name="generator")
    manager_discriminator = tf.train.CheckpointManager(checkpoint_discriminator, directory='model_weight/discriminator',max_to_keep=10,checkpoint_name="discriminator")


    optimizer = tf.keras.optimizers.Adam(1e-7)

    encoder_rec_loss, encoder_adv_loss, dis_loss = training(311, 50, 120)

    plt.plot(encoder_rec_loss)
    plt.title("the encoder_recontruction_loss")
    plt.savefig("result_image/encoder_reconstruction_loss.jpg")
    plt.close()

    plt.plot(encoder_adv_loss)
    plt.title("the encoder_adv_loss")
    plt.savefig("result_image/encoder_adv_loss.jpg")
    plt.close()

    # plt.plot(encoder_percetual_loss)
    # plt.title("the encoder_percetual_loss")
    # plt.savefig("highfacegan_result_image/encoder_percetual_loss.jpg")
    # plt.close()

    plt.plot(dis_loss)
    plt.plot(encoder_adv_loss)
    plt.title("the adv_loss")
    plt.legend(["dis_adv","gen_adv"],loc="upper right")
    plt.savefig("result_image/adv_loss.jpg")
    plt.close()

