from build_model import *
from prepare_data import *
from loss import *

def discriminator_train_step(real_low, real_high):
    with tf.GradientTape() as tape:
        code = encoder(real_low)
        fake_high = generator(code)
        real_high = discriminator(real_high)
        fake_high = discriminator(fake_high)

        real_local = discriminator_decoder(real_high)
        fake_local = discriminator_decoder(fake_high)
        dis_local_loss = Local_discriminator_loss(real_local, fake_local)

    grads = tape.gradient(dis_local_loss, discriminator_decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator_decoder.trainable_variables))
    return dis_local_loss

def training(epochs,batch_size,batch_num):
    dis_local_loss_avg = []
    for epoch in range(1, epochs):
        dis_local_loss_epoch = []
        start = time.time()
        train_path, _ = load_path()
        for batch in range(batch_num):
            real_low, real_high = load_image(get_batch_data(train_path, batch, batch_size), train=True)
            real_high, real_low = real_high.reshape(-1,64,64,1), real_low.reshape(-1, 64, 64, 1)
            dis_local_loss = discriminator_train_step(real_low, real_high)
            dis_local_loss_epoch.append(dis_local_loss)

        dis_local_loss_avg.append(np.mean(dis_local_loss_epoch))
        print(f"the epoch is {epoch}")
        print(f"the dis_loss is {dis_local_loss_avg[-1]}")
        print("the spend time is : %s seconds " % (time.time() - start))
        manager_discriminator_decoder.save(checkpoint_number=epoch+400)
        draw_samples(epoch, train=True)
        draw_samples(epoch, train=False)
    return dis_local_loss_avg


def draw_samples(epoch,train=True,path="result/instance_highfacegan_result_image/"):
    if train:
        train_path, _ = load_path()
        real_low, real_high = load_image(get_batch_data(train_path, 0, 10),train=True)
    else:
        _, test_path = load_path()
        real_low, real_high = load_image(get_batch_data(test_path, 100, 10),train=False)

    real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)

    code = encoder(real_low)
    fake_high = generator(code)

    real = discriminator(real_high)
    fake = discriminator(fake_high)

    real_local = discriminator_decoder(real)
    fake_local = discriminator_decoder(fake)
    # print(real_local.shape, fake_local.shape)

    plt.subplots(figsize=(10, 4))
    for i in range(10):
        plt.subplot(4, 10, i + 1)
        plt.axis("off")
        plt.imshow(real_high[i].reshape(64,64), cmap="gray")
        plt.subplot(4, 10, i + 11)
        plt.axis("off")
        plt.imshow(real_local[i][:, :, 0], cmap="gray")
        plt.subplot(4, 10, i + 21)
        plt.axis("off")
        plt.imshow(tf.reshape(fake_high[i],[64,64]), cmap="gray")
        plt.subplot(4, 10, i + 31)
        plt.axis("off")
        plt.imshow(fake_local[i][:, :, 0], cmap="gray")
    if train:
        plt.savefig(path + f"local_train_produce_image_{epoch}epochs")
    else:
        plt.savefig(path + f"local_test_produce_image_{epoch}epochs")
    plt.close()



if __name__ == "__main__":
    encoder = encoder()
    generator = generator()
    discriminator = discriminator()
    discriminator_decoder = discriminator_decoder()

    #set checkpoint
    checkpoint_encoder = tf.train.Checkpoint(encoder)
    checkpoint_generator = tf.train.Checkpoint(generator)
    checkpoint_discriminator = tf.train.Checkpoint(discriminator)
    checkpoint_discriminator_decoder = tf.train.Checkpoint(discriminator_decoder)
    checkpoint_encoder.restore("model_weight/instance_encoder/encoder-501")
    checkpoint_generator.restore("model_weight/instance_generator/generator-501")
    checkpoint_discriminator.restore("model_weight/instance_discriminator/discriminator-500")

    manager_discriminator_decoder = tf.train.CheckpointManager(checkpoint_discriminator_decoder, directory='model_weight/discriminator_decoder', max_to_keep=15,checkpoint_name="discriminator_decoder")


    optimizer = tf.keras.optimizers.Adam(1e-5)

    dis_decoder_loss = training(60, 50, 120)

    plt.plot(dis_decoder_loss)
    plt.title("the dis_decoder_loss_loss")
    plt.savefig("result/instance_highfacegan_result_image/dis_decoder_loss")
    plt.close()





