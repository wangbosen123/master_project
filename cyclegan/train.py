import matplotlib.pyplot as plt

from build_model import *
from deal_data import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from loss import *
import time


class CycleGAN_training():
    def __init__(self, epochs, batch_num, batch_size):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.generator_ab = build_generator()
        self.generator_ba = build_generator()
        self.discriminator_a = build_discriminator()
        self.discriminator_b = build_discriminator()
        self.optimizers = tf.keras.optimizers.Adam(5e-5)
        #set the checkpoint
        self.checkpoint_generator_ab = tf.train.Checkpoint(self.generator_ab)
        self.checkpoint_generator_ba = tf.train.Checkpoint(self.generator_ba)
        self.checkpoint_discriminator_a = tf.train.Checkpoint(self.discriminator_a)
        self.checkpoint_discriminator_b = tf.train.Checkpoint(self.discriminator_b)
        # self.checkpoint_generator_ab.restore("model_weight/GAB/generator_ab-6")
        # self.checkpoint_generator_ba.restore("model_weight/GBA/generator_ba-6")
        # self.checkpoint_discriminator_a.restore("model_weight/DA/discriminator_a-6")
        # self.checkpoint_discriminator_b.restore("model_weight/DB/discriminator_b-6")
        self.manager_generator_ab = tf.train.CheckpointManager(self.checkpoint_generator_ab, directory='model_weight/GAB', max_to_keep=15, checkpoint_name="generator_ab")
        self.manager_generator_ba = tf.train.CheckpointManager(self.checkpoint_generator_ba, directory='model_weight/GBA', max_to_keep=15, checkpoint_name="generator_ba")
        self.manager_discriminator_a = tf.train.CheckpointManager(self.checkpoint_discriminator_a,directory='model_weight/DA',max_to_keep=15, checkpoint_name="discriminator_a")
        self.manager_discriminator_b = tf.train.CheckpointManager(self.checkpoint_discriminator_b,directory='model_weight/DB',max_to_keep=15, checkpoint_name="discriminator_b")


    def generator_train_step(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            # a to b
            gen_img_b = self.generator_ab(inputs[:,:,:,0])
            re_gen_img_a = self.generator_ba(gen_img_b)
            same_b = self.generator_ab(inputs[:,:,:,1])

            re_gen_img_a = tf.reshape(re_gen_img_a, [-1, 128, 128])
            same_b = tf.reshape(same_b, [-1,128,128])
            fake_b = self.discriminator_b(gen_img_b)
            adv_loss_b = generator_loss(fake_b)
            cycle_forward_loss = cycle_consistency_loss(inputs[:,:,:,0], re_gen_img_a)
            identy_loss_b = identity_loss(inputs[:,:,:,1], same_b)


            # b to a
            gen_img_a = self.generator_ba(inputs[:,:,:,1])
            re_gen_img_b = self.generator_ab(gen_img_a)
            same_a = self.generator_ba(inputs[:,:,:,0])

            re_gen_img_b = tf.reshape(re_gen_img_b, [-1, 128, 128])
            same_a = tf.reshape(same_a, [-1,128,128])
            fake_a = self.discriminator_a(gen_img_a)
            adv_loss_a = generator_loss(fake_a)
            cycle_backward_loss = cycle_consistency_loss(inputs[:,:,:,1], re_gen_img_b)
            identy_loss_a = identity_loss(inputs[:,:,:,0], same_a)

            #compute the loss
            cycle_loss = 10*(cycle_forward_loss + cycle_backward_loss)
            identy_loss = 5*(identy_loss_a + identy_loss_b)
            total_loss_gab = adv_loss_b + 10*cycle_backward_loss + 5*identy_loss_b
            total_loss_gba = adv_loss_a + 10*cycle_forward_loss + 5*identy_loss_a

        grads_gab = tape.gradient(total_loss_gab, self.generator_ab.trainable_variables)
        grads_gba = tape.gradient(total_loss_gba, self.generator_ba.trainable_variables)
        self.optimizers.apply_gradients(zip(grads_gab, self.generator_ab.trainable_variables))
        self.optimizers.apply_gradients(zip(grads_gba, self.generator_ba.trainable_variables))

        return adv_loss_a, adv_loss_b, cycle_loss, identy_loss


    def d_train_step(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            # a to b
            gen_img_b = self.generator_ab(inputs[:, :, :, 0])
            fake_b = self.discriminator_b(gen_img_b)
            real_b = self.discriminator_b(inputs[:,:,:,1])
            dis_loss_b = 0.5 * discriminator_loss(real_b, fake_b)

            # b to a
            gen_img_a = self.generator_ba(inputs[:, :, :, 1])
            fake_a = self.discriminator_a(gen_img_a)
            real_a = self.discriminator_a(inputs[:,:,:,0])
            dis_loss_a = 0.5 * discriminator_loss(real_a, fake_a)

            adv_loss = dis_loss_a + dis_loss_b

        grads_a = tape.gradient(dis_loss_a, self.discriminator_a.trainable_variables)
        grads_b = tape.gradient(dis_loss_b, self.discriminator_b.trainable_variables)
        self.optimizers.apply_gradients(zip(grads_a, self.discriminator_a.trainable_variables))
        self.optimizers.apply_gradients(zip(grads_b, self.discriminator_b.trainable_variables))
        return adv_loss


    def training(self):
        gab_adv_loss_avg = []
        gba_adv_loss_avg = []
        cyc_loss_avg = []
        id_loss_avg = []
        discriminator_loss_avg = []
        for epoch in range(1, self.epochs):
            start = time.time()
            gab_adv_loss_epoch = []
            gba_adv_loss_epoch = []
            cyc_loss_epoch = []
            id_loss_epoch = []
            discriminator_loss_epoch = []
            domain_a, domain_b, _ = load_path()
            for batch in range(self.batch_num):
                domain_a_image = load_image(get_batch_data(domain_a, batch, self.batch_size))
                domain_b_image = load_image(get_batch_data(domain_b, batch, self.batch_size))
                domain_a_image, domain_b_image = domain_a_image.reshape(-1, 128, 128, 1), domain_b_image.reshape(-1, 128, 128, 1)
                data = np.concatenate((domain_a_image, domain_b_image), axis=-1)


                # for i in range(2):
                gba_adv_loss, gab_adv_loss, cyc_loss, id_loss = self.generator_train_step(data)
                gba_adv_loss_epoch.append(gba_adv_loss)
                gab_adv_loss_epoch.append(gab_adv_loss)
                cyc_loss_epoch.append(cyc_loss)
                id_loss_epoch.append(id_loss)

                for i in range(2):
                    d_adv_loss = self.d_train_step(data)
                    discriminator_loss_epoch.append(d_adv_loss)

            gab_adv_loss_avg.append(np.mean(gab_adv_loss_epoch))
            gba_adv_loss_avg.append(np.mean(gba_adv_loss_epoch))
            cyc_loss_avg.append(np.mean(cyc_loss_epoch))
            id_loss_avg.append(np.mean(id_loss_epoch))
            discriminator_loss_avg.append(np.mean(discriminator_loss_epoch))
            print(f"the epoch is {epoch}")
            print(f"the cycle consistent loss is {cyc_loss_avg[-1]}")
            print(f"the identity loss is {id_loss_avg[-1]}")
            print(f"the generator a_b adv loss is {gab_adv_loss_avg[-1]}")
            print(f"the generator b_a adv loss is {gba_adv_loss_avg[-1]}")
            print(f"the discriminator adv loss is {discriminator_loss_avg[-1]}")
            print(f"the spend time is {time.time() - start} second")
            self.draw_samples(epoch)
            if epoch % 10 == 0 :
                self.manager_generator_ab.save(checkpoint_number=epoch)
                self.manager_generator_ba.save(checkpoint_number=epoch)
                self.manager_discriminator_a.save(checkpoint_number=epoch)
                self.manager_discriminator_b.save(checkpoint_number=epoch)
        return gab_adv_loss_avg, gba_adv_loss_avg, cyc_loss_avg, id_loss_avg, discriminator_loss_avg

    def draw_samples(self, epoch):
        domain_a, domain_b, domain_a_test = load_path()
        domain_a = load_image(get_batch_data(domain_a, 0, 10))
        domain_b = load_image(get_batch_data(domain_b, 0, 10))
        domain_a_test = load_image(get_batch_data(domain_a_test, 0, 10))
        domain_a, domain_b, domain_a_test = domain_a.reshape(-1, 128, 128, 1), domain_b.reshape(-1, 128, 128, 1), domain_a_test.reshape(-1,128,128,1)

        gen_image_a = self.generator_ba(domain_b)
        gen_image_b = self.generator_ab(domain_a)
        gen_image_b_test = self.generator_ab(domain_a_test)
        gen_image_a, gen_image_b, gen_image_b_test = tf.reshape(gen_image_a, [-1,128,128]), tf.reshape(gen_image_b, [-1,128,128]), tf.reshape(gen_image_b_test, [-1,128,128])

        plt.subplots(figsize=(15,4))
        for i in range(10):
            plt.subplot(2,10,i+1)
            plt.axis("off")
            plt.imshow(domain_a[i], cmap="gray")
            plt.subplot(2, 10, i + 11)
            plt.axis("off")
            plt.imshow(gen_image_b[i], cmap="gray")
        plt.savefig(f"result_image/domain_a_to_domain_b_{epoch}")
        plt.close()

        plt.subplots(figsize=(15, 4))
        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.axis("off")
            plt.imshow(domain_b[i], cmap="gray")
            plt.subplot(2, 10, i + 11)
            plt.axis("off")
            plt.imshow(gen_image_a[i], cmap="gray")
        plt.savefig(f"result_image/omain_b_to_domain_a_{epoch}")
        plt.close()

        plt.subplots(figsize=(15, 4))
        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.axis("off")
            plt.imshow(domain_a_test[i], cmap="gray")
            plt.subplot(2, 10, i + 11)
            plt.axis("off")
            plt.imshow(gen_image_b_test[i], cmap="gray")
        plt.savefig(f"result_image/test_domain_a_to_domain_b_{epoch}")
        plt.close()

if __name__ == "__main__":
    cycleGAN = CycleGAN_training(60,132,5)
    gab_adv_loss_avg, gba_adv_loss_avg, cyc_loss_avg, id_loss_avg, discriminator_loss_avg = cycleGAN.training()


    plt.plot(cyc_loss_avg)
    plt.title("generator_cyc_loss")
    plt.savefig("result_image/generator_cyc_loss")
    plt.close()

    plt.plot(id_loss_avg)
    plt.title("Identity_loss")
    plt.savefig("result_image/Identity_loss")
    plt.close()

    plt.plot(gab_adv_loss_avg)
    plt.plot(gba_adv_loss_avg)
    plt.plot(discriminator_loss_avg)
    plt.title("adv_loss")
    plt.legend(["GAB_adv", "GBA_adv", "D_adv"], loc="upper right")
    plt.savefig("result_image/adv_loss")
    plt.close()









