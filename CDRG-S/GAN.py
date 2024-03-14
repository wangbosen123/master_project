import numpy as np
import cv2
from build_model import *
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.cm as cm

class PatchGAN():
    def __init__(self, epochs, batch_num, batch_size):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.g_opti = tf.keras.optimizers.Adam(1e-5)
        self.d_opti = tf.keras.optimizers.Adam(1e-5)

        #set the model
        self.encoder = encoder()
        self.reg = regression()
        self.generator = generator()
        self.discriminator = discriminator()
        self.encoder.load_weights('weights/encoder')
        self.generator.load_weights('weights/generator2')
        self.reg.load_weights('weights/reg_x_cls_REG')
        self.discriminator.load_weights('weights/discriminator2')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        #set the data path
        self.train_path, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        path_AR_syn_train = '/disk2/bosen/Datasets/AR_train/'
        path_AR_syn_test = '/disk2/bosen/Datasets/AR_test/'
        path_AR_real_train = "/disk2/bosen/Datasets/AR_original_alignment_train90/"
        train_path = []
        test_path1, test_path2, test_path3 = [], [], []
        ID = [f'ID{i}' for i in range(1, 91)]

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2070:
                train_path.append(path_celeba + filename)
            if num < 21:
                test_path3.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                if num == 21:
                    test_path2.append(path_AR_syn_train + id + '/' + filename)

        for count, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_AR_real_train + id)):
                if '-1-0' in filename or '-1-1' in filename or '-1-2' in filename:
                    train_path.append(path_AR_real_train + id + '/' + filename)

        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if num == 1:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)


        train_path, test_path1, test_path2, test_path3 = np.array(train_path), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        np.random.shuffle(train_path)
        return train_path, test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size):
        train_images, ground_truth = [], []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        for path in train_data:
            image = cv2.imread(path, 0) / 255
            if "AR" in path:
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            train_images.append(image), train_images.append(low1_image), train_images.append(low2_image), train_images.append(low3_image)
            ground_truth.append(image), ground_truth.append(image), ground_truth.append(image), ground_truth.append(image)

        ground_truth = np.array(ground_truth).reshape(-1, 64, 64, 1)
        train_images = np.array(train_images).reshape(-1, 64, 64, 1)
        return ground_truth, train_images

    def style_loss(self, real, fake):
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = self.feature_extraction(real)
        fake_feature = self.feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def g_train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_images = self.generator(zreg)
            fake_score = self.discriminator(gen_images)


            g_loss = tf.reduce_mean(tf.square(fake_score - 1))
            image_loss = 100 * tf.reduce_mean(tf.square(high_images - gen_images))
            style_loss = 100 * self.style_loss(high_images, gen_images)
            total_loss = g_loss + image_loss + style_loss

        if train:
            grads = tape.gradient(total_loss, self.generator.trainable_variables)
            self.g_opti.apply_gradients(zip(grads, self.generator.trainable_variables))
        return image_loss, style_loss, g_loss

    def d_train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_image = self.generator(zreg)
            real_score = self.discriminator(high_images)
            fake_score = self.discriminator(gen_image)
            d_loss = (tf.reduce_mean(tf.square(real_score - 1)) + tf.reduce_mean(tf.square(fake_score)))

        if train:
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opti.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return d_loss

    def training(self):
        image_loss_epoch = []
        style_loss_epoch = []
        g_loss_epoch = []
        d_loss_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            g_loss_batch = []
            d_loss_batch = []

            for batch in range(self.batch_num):
                high_images, low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                # for i in range(2):
                #     d_loss = self.d_train_step(low_images, high_images, train=True)
                # d_loss_batch.append(d_loss)

                image_loss, style_loss, g_loss = self.g_train_step(low_images, high_images, train=True)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                g_loss_batch.append(g_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            g_loss_epoch.append(np.mean(g_loss_batch))
            d_loss_epoch.append(np.mean(d_loss_batch))
            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the g_loss is {g_loss_epoch[-1]}')
            print(f'the d_loss is {d_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.generator.save_weights('weights/generator4')
            self.discriminator.save_weights('weights/discriminator4')

            if (epoch == 1) or (epoch%5==0):
                self.plot_image(epoch, self.test_path1, data_name='test_21ID_normal', reg=True)
                self.plot_image(epoch, self.test_path1, data_name='test_21ID_normal', reg=False)


            plt.plot(image_loss_epoch, label='Image Loss')
            plt.title('Image Loss')
            plt.savefig('result/GAN/image_loss4')
            plt.close()

            plt.plot(style_loss_epoch, label='Style Loss')
            plt.title('Style Loss')
            plt.savefig('result/GAN/style_loss4')
            plt.close()

            plt.plot(g_loss_epoch, label='G Loss')
            plt.title('G Loss')
            plt.savefig('result/GAN/g_loss4')
            plt.close()

            plt.plot(d_loss_epoch, label='D Loss')
            plt.title('D Loss')
            plt.savefig('result/GAN/d_loss4')
            plt.close()

            plt.plot(g_loss_epoch, label='G Loss')
            plt.plot(d_loss_epoch, label='D Loss')
            plt.legend(['G Loss', 'D Loss'], loc='upper right')
            plt.savefig('result/GAN/adv_loss4')
            plt.close()

    def plot_image(self, epoch, path, data_name, reg=True):
        plt.subplots(figsize=(7, 8))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        PSNR = [[] for i in range(3)]
        for num, filename in enumerate(path):
            image = cv2.imread(filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))

            if reg:
                _, _, z = self.reg(z)
                _, _, z1 = self.reg(z1)
                _, _, z2 = self.reg(z2)
                _, _, z3 = self.reg(z3)

            gen_image = self.generator(z)
            gen1_image = self.generator(z1)
            gen2_image = self.generator(z2)
            gen3_image = self.generator(z3)

            plt.subplot(8, 7, count + 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')

            plt.subplot(8, 7, count + 8)
            plt.axis('off')
            plt.imshow(tf.reshape(gen_image, [64, 64]), cmap='gray')

            plt.subplot(8, 7, count + 15)
            plt.axis('off')
            plt.imshow(low1_image, cmap='gray')

            plt.subplot(8, 7, count + 22)
            plt.axis('off')
            plt.imshow(tf.reshape(gen1_image, [64, 64]), cmap='gray')

            plt.subplot(8, 7, count + 29)
            plt.axis('off')
            plt.imshow(low2_image, cmap='gray')

            plt.subplot(8, 7, count + 36)
            plt.axis('off')
            plt.imshow(tf.reshape(gen2_image, [64, 64]), cmap='gray')

            plt.subplot(8, 7, count + 43)
            plt.axis('off')
            plt.imshow(low3_image, cmap='gray')

            plt.subplot(8, 7, count + 50)
            plt.axis('off')
            plt.imshow(tf.reshape(gen3_image, [64, 64]), cmap='gray')
            count += 1
            PSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(gen1_image, dtype=tf.float32), max_val=1)[0])
            PSNR[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(gen2_image, dtype=tf.float32), max_val=1)[0])
            PSNR[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(gen3_image, dtype=tf.float32), max_val=1)[0])

            if (num+1) % 7 == 0:
                if reg:
                    plt.savefig(f'result/GAN/after_reg_{data_name}_{epoch}_{num+1}image')
                else:
                    plt.savefig(f'result/GAN/before_reg_{data_name}_{epoch}_{num+1}image')

                plt.close()
                plt.subplots(figsize=(7, 8))
                plt.subplots_adjust(hspace=0, wspace=0)
                count = 0
        print(f'2 ratio PSNR is {np.mean(PSNR[0])}')
        print(f'4 ratio PSNR is {np.mean(PSNR[1])}')
        print(f'8 ratio PSNR is {np.mean(PSNR[2])}')


    def validate_G_trained_well(self, id_index, train=True):
        if train:
            path = '/home/bosen/PycharmProjects/Datasets/AR_train/'
            target_image = cv2.resize(cv2.imread(f'/home/bosen/PycharmProjects/Datasets/AR_train/ID{id_index}/1_train.jpg', 0) / 255, (64, 64), cv2.INTER_CUBIC)
            target_low1_image = cv2.resize(cv2.resize(target_image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            target_low2_image = cv2.resize(cv2.resize(target_image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            target_low3_image = cv2.resize(cv2.resize(target_image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            ids = [f'ID{i}' for i in range(1, 91)]
            face_low1 = [[np.zeros((64, 64)), np.zeros((64, 64))] for i in range(90)]
            face_low2 = [[np.zeros((64, 64)), np.zeros((64, 64))] for i in range(90)]
            face_low3 = [[np.zeros((64, 64)), np.zeros((64, 64))] for i in range(90)]
            id_error = [[1e20 for i in range(3)] for i in range(90)]

        else:
            path = '/home/bosen/PycharmProjects/Datasets/AR_test/'
            target_image = cv2.resize(cv2.imread(f'/home/bosen/PycharmProjects/Datasets/AR_test/ID0{id_index}/1_test.jpg', 0) / 255, (64, 64), cv2.INTER_CUBIC)
            target_low1_image = cv2.resize(cv2.resize(target_image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            target_low2_image = cv2.resize(cv2.resize(target_image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            target_low3_image = cv2.resize(cv2.resize(target_image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            ids = [f'ID0{i}' if i < 10 else f'ID{i}' for i in range(1, 22)]
            face_low1 = [[np.zeros((64, 64)), np.zeros((64, 64))] for i in range(21)]
            face_low2 = [[np.zeros((64, 64)), np.zeros((64, 64))] for i in range(21)]
            face_low3 = [[np.zeros((64, 64)), np.zeros((64, 64))] for i in range(21)]
            id_error = [[1e20 for i in range(3)] for i in range(21)]


        for num_id, id in enumerate(ids):
            for num_image, filename in enumerate(os.listdir(path + id)):
                image = cv2.resize(cv2.imread(path + id + '/' + filename, 0)/255, (64, 64), cv2.INTER_CUBIC)
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                low1_error = tf.reduce_mean(tf.square(target_low1_image - low1_image))
                low2_error = tf.reduce_mean(tf.square(target_low2_image - low2_image))
                low3_error = tf.reduce_mean(tf.square(target_low3_image - low3_image))

                if low1_error < id_error[num_id][0]:
                    id_error[num_id][0] = low1_error
                    face_low1[num_id][0] = image
                    face_low1[num_id][1] = low1_image
                if low2_error < id_error[num_id][0]:
                    id_error[num_id][1] = low2_error
                    face_low2[num_id][0] = image
                    face_low2[num_id][1] = low2_image
                if low3_error < id_error[num_id][0]:
                    id_error[num_id][2] = low3_error
                    face_low3[num_id][0] = image
                    face_low3[num_id][1] = low3_image

        id_error = np.array(id_error)
        id_error_low1 = list(id_error[:, 0])
        id_error_low2 = list(id_error[:, 1])
        id_error_low3 = list(id_error[:, 2])

        id_error_low1[id_error_low1.index(min(id_error_low1))] = 1e20
        id_error_low2[id_error_low2.index(min(id_error_low2))] = 1e20
        id_error_low3[id_error_low3.index(min(id_error_low3))] = 1e20

        nn_face_low1 = face_low1[id_error_low1.index(min(id_error_low1))]
        nn_face_low2 = face_low2[id_error_low2.index(min(id_error_low2))]
        nn_face_low3 = face_low3[id_error_low3.index(min(id_error_low3))]

        plt.subplots(figsize=(4, 7))
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.subplot(7, 4, 1)
        plt.axis('off')
        plt.imshow(target_image, cmap='gray')
        plt.subplot(7, 4, 2)
        plt.axis('off')
        plt.imshow(nn_face_low1[0], cmap='gray')
        plt.subplot(7, 4, 3)
        plt.axis('off')
        plt.imshow(nn_face_low2[0], cmap='gray')
        plt.subplot(7, 4, 4)
        plt.axis('off')
        plt.imshow(nn_face_low3[0], cmap='gray')

        plt.subplot(7, 4, 5)
        plt.axis('off')
        plt.imshow(target_low1_image, cmap='gray')
        plt.subplot(7, 4, 6)
        plt.axis('off')
        plt.imshow(nn_face_low1[1], cmap='gray')
        plt.subplot(7, 4, 7)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        plt.subplot(7, 4, 8)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')

        z = self.encoder(target_low1_image.reshape(1, 64, 64, 1))
        zd, _, _ = self.ztozg(z)
        syn_image = self.generator(zd)
        plt.subplot(7, 4, 9)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        z = self.encoder(nn_face_low1[1].reshape(1, 64, 64, 1))
        zd, _, _ = self.ztozg(z)
        syn_image = self.generator(zd)
        plt.subplot(7, 4, 10)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        plt.subplot(7, 4, 11)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        plt.subplot(7, 4, 12)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')

        plt.subplot(7, 4, 13)
        plt.axis('off')
        plt.imshow(target_low2_image, cmap='gray')
        plt.subplot(7, 4, 14)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        plt.subplot(7, 4, 15)
        plt.axis('off')
        plt.imshow(nn_face_low2[1], cmap='gray')
        plt.subplot(7, 4, 16)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')

        z = self.encoder(target_low2_image.reshape(1, 64, 64, 1))
        zd, _, _ = self.ztozg(z)
        syn_image = self.generator(zd)
        plt.subplot(7, 4, 17)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        plt.subplot(7, 4, 18)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        z = self.encoder(nn_face_low2[1].reshape(1, 64, 64, 1))
        zd, _, _ = self.ztozg(z)
        syn_image = self.generator(zd)
        plt.subplot(7, 4, 19)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        plt.subplot(7, 4, 20)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')

        plt.subplot(7, 4, 21)
        plt.axis('off')
        plt.imshow(target_low3_image, cmap='gray')
        plt.subplot(7, 4, 22)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        plt.subplot(7, 4, 23)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        plt.subplot(7, 4, 24)
        plt.axis('off')
        plt.imshow(nn_face_low3[1], cmap='gray')

        z = self.encoder(target_low3_image.reshape(1, 64, 64, 1))
        zd, _, _ = self.ztozg(z)
        syn_image = self.generator(zd)
        plt.subplot(7, 4, 25)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        plt.subplot(7, 4, 26)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        plt.subplot(7, 4, 27)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        z = self.encoder(nn_face_low3[1].reshape(1, 64, 64, 1))
        zd, _, _ = self.ztozg(z)
        syn_image = self.generator(zd)
        plt.subplot(7, 4, 28)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        if train:
            plt.savefig(f'result/PatchGAN/train_id{id_index}_similarity_syn')
            plt.close()
        else:
            plt.savefig(f'result/PatchGAN/test_id{id_index}_similarity_syn')
            plt.close()

    def validate_D_trained_well(self):
        path_test = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        real_data_value, syn1_data_value, syn2_data_value, syn3_data_value = [], [], [], []
        for id in os.listdir(path_test):
            for filename in os.listdir(path_test + id):
                image = cv2.resize(cv2.imread(path_test + id + '/' + filename, 0) / 255, (64, 64), cv2.INTER_CUBIC)
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                z1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                z2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                z3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
                zg1, _, _ = self.ztozg(z1)
                zg2, _, _ = self.ztozg(z2)
                zg3, _, _ = self.ztozg(z3)
                syn1_image = self.generator(zg1)
                syn2_image = self.generator(zg2)
                syn3_image = self.generator(zg3)

                real_value = self.discriminator(image.reshape(1, 64, 64, 1))
                syn1_value = self.discriminator(syn1_image)
                syn2_value = self.discriminator(syn2_image)
                syn3_value = self.discriminator(syn3_image)

                real_data_value.append(tf.reduce_mean(real_value))
                syn1_data_value.append(tf.reduce_mean(syn1_value))
                syn2_data_value.append(tf.reduce_mean(syn2_value))
                syn3_data_value.append(tf.reduce_mean(syn3_value))

        real_data_value, syn1_data_value, syn2_data_value, syn3_data_value = np.array(real_data_value), np.array(syn1_data_value), np.array(syn2_data_value), np.array(syn3_data_value)
        plt.hist(real_data_value)
        plt.hist(syn1_data_value, alpha=0.8)
        plt.legend(['real_data_mean_value', '2_ratio_synimage_mean_value'], loc='upper left')
        plt.savefig('result/PatchGAN/Discriminator_distribution_real_vs_2rationsyn')
        plt.close()

        plt.hist(real_data_value)
        plt.hist(syn2_data_value, alpha=0.6)
        plt.legend(['real_data_mean_value', '4_ratio_synimage_mean_value'], loc='upper left')
        plt.savefig('result/PatchGAN/Discriminator_distribution_real_vs_4rationsyn')
        plt.close()

        plt.hist(real_data_value)
        plt.hist(syn3_data_value, alpha=0.4)
        plt.legend(['real_data_mean_value', '8_ratio_synimage_mean_value'], loc='upper left')
        plt.savefig('result/PatchGAN/Discriminator_distribution_real_vs_8rationsyn')
        plt.close()



if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    AE = PatchGAN(epochs=50, batch_num=69, batch_size=60)
    AE.training()

