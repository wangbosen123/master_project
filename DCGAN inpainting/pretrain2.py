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

def discriminator_train_step(code, train_data):
    with tf.GradientTape() as tape:
        fake = generator(code)
        fake = discriminator(fake)
        real = discriminator(train_data)
        real_fake_loss = discriminator_loss(real,fake)
        dis_loss = 0.5 * real_fake_loss
    grads = tape.gradient(dis_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return dis_loss



def encoder_train_step(code):
    with tf.GradientTape() as tape:
        fake_image = generator(code)
        fake = discriminator(fake_image)
        real_fake_loss = generator_loss(fake)
        encoder_loss = real_fake_loss
    grads = tape.gradient(encoder_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return real_fake_loss


def training(epochs,batch_size,batch_num):
    encoder_adv_loss_avg = []
    dis_loss_avg = []
    for epoch in range(1, epochs+1):

        encoder_adv_loss_epoch = []
        dis_loss_epoch = []
        start = time.time()
        for batch in range(batch_num):
            code = tf.random.uniform([batch_size, 100], minval=-1, maxval=1)
            train_data = load_image(get_batch_data(load_path(train=True), batch, batch_size))
            train_data = train_data.reshape(-1,64,64,1)
            dis_loss = discriminator_train_step(code, train_data)
            dis_loss_epoch.append(dis_loss)
            encoder_adv_loss = encoder_train_step(code)
            encoder_adv_loss_epoch.append(encoder_adv_loss)


        encoder_adv_loss_avg.append(np.mean(encoder_adv_loss_epoch))
        dis_loss_avg.append(np.mean(dis_loss_epoch))
        print(f"the epoch is {epoch}")
        print(f"the generator_adv_loss is {encoder_adv_loss_avg[-1]}")
        print(f"the dis_loss is {dis_loss_avg[-1]}")
        print("the spend time is : %s seconds " % (time.time() - start))

        if epoch % 10 == 0:
            manager_generator.save(checkpoint_number=epoch)
            manager_discriminator.save(checkpoint_number=epoch)
            draw_samples(epoch, train=True)
            draw_samples(epoch, train=False)

    return encoder_adv_loss_avg, dis_loss_avg

def draw_samples(epoch,train=True,path="result_image/"):
    code = tf.random.normal([10,100], mean=0, stddev=1)
    train_data = load_image(get_batch_data(load_path(train=False), 0, 10))

    fake = generator(code)
    fake = tf.reshape(fake, [-1, 64, 64])

    ax, fig = plt.subplots(figsize=(15, 4))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.axis("off")
        plt.imshow(train_data[i], cmap="gray")
        plt.subplot(2, 10, i + 11)
        plt.axis("off")
        plt.imshow(fake[i], cmap="gray")


    plt.savefig(path + f"onlyrealfake_produce_image_{epoch}epochs")
    plt.close()






if __name__ == "__main__":
    generator = build_model.generator()
    discriminator = build_model.discriminator()

    #set checkpoint
    checkpoint_generator = tf.train.Checkpoint(generator)
    checkpoint_discriminator = tf.train.Checkpoint(discriminator)
    checkpoint_generator.restore("model_weight/generator/generator_899_weights")
    checkpoint_discriminator.restore("model_weight/discriminator/discriminator_899_weights")
    manager_generator = tf.train.CheckpointManager(checkpoint_generator, directory='model_weight/generator', max_to_keep=15,checkpoint_name="generator")
    manager_discriminator = tf.train.CheckpointManager(checkpoint_discriminator, directory='model_weight/discriminator',max_to_keep=15,checkpoint_name="discriminator")


    optimizer = tf.keras.optimizers.Adam(1e-7)

    encoder_adv_loss, dis_loss = training(300, 50, 100)



    plt.plot(encoder_adv_loss)
    plt.title("the generator_adv_loss")
    plt.savefig("result_image/generator_adv_loss.jpg")
    plt.close()


    plt.plot(dis_loss)
    plt.plot(encoder_adv_loss)
    plt.title("the adv_loss")
    plt.legend(["dis_adv", "gen_adv"], loc="upper right")
    plt.savefig("result_image/adv_loss.jpg")
    plt.close()