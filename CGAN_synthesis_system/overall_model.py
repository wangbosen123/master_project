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


class AE():
    def __init__(self, epochs, batch_num, batch_size):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        #set the model
        self.encoder = normal_encoder()
        self.ztozd = ZtoZd()
        self.decoder = decoder()
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozd.load_weights('model_weight/AE_ztozd')
        self.decoder.load_weights('model_weight/AE_decoder')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        #set the data path
        self.train_path, self.train_label, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.train_label.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = "/home/bosen/PycharmProjects/SRGAN_learning_based_inversion/celeba_train/"
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
        path_AR_real_train = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_train90/"
        # path_AR_real_test = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_test21/"
        train_path, train_label = [], []
        test_path1, test_path2, test_path3 = [], [], []
        ID = [f'ID{i}' for i in range(1, 91)]


        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2070:
                train_path.append(path_celeba + filename)
                train_label.append(tf.one_hot(90, 91))
            if num < 21:
                test_path3.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                    train_label.append(tf.one_hot(int(id[2:])-1, 91))
                if num == 21:
                    test_path2.append(path_AR_syn_train + id + '/' + filename)


        for count, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_AR_real_train + id)):
                if '-1-0' in filename or '-1-1' in filename or '-1-2' in filename:
                    train_path.append(path_AR_real_train + id + '/' + filename)
                    train_label.append(tf.one_hot(int(id[2:])-1, 91))

                # if '-1-0' in filename and count < 21:
                #     test_path2.append(path_AR_real_train + id + '/' + filename)

        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if '11_test' in filename:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)

        train_path, train_label, test_path1, test_path2, test_path3 = np.array(train_path), np.array(train_label), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        train_data = list(zip(train_path, train_label))
        np.random.shuffle(train_data)
        train_data = list(zip(*train_data))
        return np.array(train_data[0]), np.array(train_data[1]), test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size, image=True):
        low_images = []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        if image:
            for path in train_data:
                image = cv2.imread(path, 0) / 255
                if "AR" in path:
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low_images.append(image)
                low_images.append(low1_image), low_images.append(low2_image), low_images.append(low3_image)

            low_images = np.array(low_images).reshape(-1, 64, 64, 1)
            return low_images
        else:
            labels = []
            for label in train_data:
                for i in range(4):
                    labels.append(label)
            labels = np.array(labels)
            return labels

    def style_loss(self, real, fake):
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = self.feature_extraction(real)
        fake_feature = self.feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def train_step(self, low_images, opti, label, train=True):
        cce = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:

            z = self.encoder(low_images)
            zd, pred = self.ztozd(z)
            gen_images = self.decoder(zd)
            image_loss = 10 * tf.reduce_mean(tf.square(low_images - gen_images))
            style_loss = 10 * self.style_loss(low_images, gen_images)
            ce_loss = 0.1 * cce(label, pred)
            acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred, axis=-1))
            total_loss = image_loss + style_loss + ce_loss

        if train:
            grads = tape.gradient(total_loss, self.ztozd.trainable_variables + self.decoder.trainable_variables)
            opti.apply_gradients(zip(grads, self.ztozd.trainable_variables + self.decoder.trainable_variables))
        return image_loss, style_loss, ce_loss, acc

    def training(self):
        image_loss_epoch = []
        style_loss_epoch = []
        ce_loss_epoch = []
        acc_epoch = []
        opti = tf.keras.optimizers.Adam(1e-4)
        for epoch in range(200, self.epochs+1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            ce_loss_batch = []
            acc_batch = []

            if epoch > 300:
                opti = tf.keras.optimizers.Adam(1e-5)

            for batch in range(self.batch_num):
                low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                train_label = self.get_batch_data(self.train_label, batch, batch_size=self.batch_size, image=False)
                image_loss, style_loss, ce_loss, acc = self.train_step(low_images, opti=opti, label=train_label, train=True)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                ce_loss_batch.append(ce_loss)
                acc_batch.append(acc)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            ce_loss_epoch.append(np.mean(ce_loss_batch))
            acc_epoch.append(np.mean(acc_batch))
            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the ce_loss is {ce_loss_epoch[-1]}')
            print(f'the acc is {acc_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            # self.encoder.save_weights('model_weight/AE_encoder')
            self.ztozd.save_weights('model_weight/AE_ztozd')
            self.decoder.save_weights('model_weight/AE_decoder')
            self.plot_image(epoch, self.test_path1, data_name='test_21ID')

            if epoch == self.epochs:
                self.plot_image(epoch, self.test_path2, data_name='train_90ID')
                self.plot_image(epoch, self.test_path3, data_name='train_celeba')

            # if epoch == 150:
            #     plt.plot(image_loss_epoch)
            #     plt.savefig('result/AE/image_loss')
            #     plt.close()
            #
            #     plt.plot(style_loss_epoch)
            #     plt.savefig('result/AE/style_loss')
            #     plt.close()
            #
            #     plt.plot(ce_loss_epoch)
            #     plt.savefig('result/AE/ce_loss')
            #     plt.close()
            #
            #     plt.plot(acc_epoch)
            #     plt.savefig('result/AE/acc')
            #     plt.close()

    def plot_image(self, epoch, path, data_name):
        plt.subplots(figsize=(7, 8))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        for num, filename in enumerate(path):
            image = cv2.imread(filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
            zd, _ = self.ztozd(z)
            zd1, _ = self.ztozd(z1)
            zd2, _ = self.ztozd(z2)
            zd3, _ = self.ztozd(z3)

            gen_image = self.decoder(zd)
            gen1_image = self.decoder(zd1)
            gen2_image = self.decoder(zd2)
            gen3_image = self.decoder(zd3)

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

            if (num+1) % 7 == 0:
                plt.savefig(f'result/AE/{data_name}_{epoch}_{num+1}image')
                plt.close()
                plt.subplots(figsize=(8, 7))
                plt.subplots_adjust(hspace=0, wspace=0)
                count = 0

    def validate_AE_trained_well(self, id_index, train=True):
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
        zd, _ = self.ztozd(z)
        syn_image = self.decoder(zd)
        plt.subplot(7, 4, 9)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        z = self.encoder(nn_face_low1[1].reshape(1, 64, 64, 1))
        zd, _ = self.ztozd(z)
        syn_image = self.decoder(zd)
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
        zd, _ = self.ztozd(z)
        syn_image = self.decoder(zd)
        plt.subplot(7, 4, 17)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        plt.subplot(7, 4, 18)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        z = self.encoder(nn_face_low2[1].reshape(1, 64, 64, 1))
        zd, _ = self.ztozd(z)
        syn_image = self.decoder(zd)
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
        zd, _ = self.ztozd(z)
        syn_image = self.decoder(zd)
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
        zd, _ = self.ztozd(z)
        syn_image = self.decoder(zd)
        plt.subplot(7, 4, 28)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        if train:
            plt.savefig(f'result/AE/train_id{id_index}_similarity_syn')
            plt.close()
        else:
            plt.savefig(f'result/AE/test_id{id_index}_similarity_syn')
            plt.close()

class PatchGAN():
    def __init__(self, epochs, batch_num, batch_size):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.g_opti = tf.keras.optimizers.Adam(1e-4)
        self.d_opti = tf.keras.optimizers.Adam(1e-4)

        #set the model
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozg.load_weights('model_weight/patch_ztozg')
        self.generator.load_weights('model_weight/patch_g')
        self.discriminator.load_weights('model_weight/patch_d')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        #set the data path
        self.train_path, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = "/home/bosen/PycharmProjects/SRGAN_learning_based_inversion/celeba_train/"
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
        path_AR_real_train = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_train90/"
        # path_AR_real_test = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_test21/"
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

                # if '-1-0' in filename and count < 21:
                #     test_path2.append(path_AR_real_train + id + '/' + filename)


        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if '11_test' in filename:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)


        train_path, test_path1, test_path2, test_path3 = np.array(train_path), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        np.random.shuffle(train_path)
        return train_path, test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size):
        high_images, low_images = [], []
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

            low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low_images.append(low1_image), low_images.append(low2_image), low_images.append(low3_image)
            high_images.append(image), high_images.append(image), high_images.append(image)

        high_images = np.array(high_images).reshape(-1, 64, 64, 1)
        low_images = np.array(low_images).reshape(-1, 64, 64, 1)
        return high_images, low_images

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
            zg, _, _ = self.ztozg(z)
            gen_images = self.generator(zg)
            fake_score = self.discriminator(gen_images)

            # fake_output = tf.concat([tf.ones((self.batch_size*3, fake_score.shape[1], fake_score.shape[2], 1), dtype="float32"), tf.zeros((self.batch_size*3, fake_score.shape[1], fake_score.shape[2], 1), dtype="float32")], axis=-1)
            # cce = tf.keras.losses.CategoricalCrossentropy()
            # g_loss = cce(fake_output, fake_score)
            # fake_gt = tf.ones_like(fake_score)
            g_loss = tf.reduce_mean(tf.square(fake_score - 1))
            image_loss = 100 * tf.reduce_mean(tf.square(high_images - gen_images))
            style_loss = 100 * self.style_loss(high_images, gen_images)
            total_loss = g_loss + image_loss + style_loss

        if train:
            grads = tape.gradient(total_loss, self.generator.trainable_variables + self.ztozg.trainable_variables)
            self.g_opti.apply_gradients(zip(grads, self.generator.trainable_variables + self.ztozg.trainable_variables))
        return image_loss, style_loss, g_loss

    def d_train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z = self.encoder(low_images)
            zg, _, _ = self.ztozg(z)
            gen_image = self.generator(zg)
            real_score = self.discriminator(high_images)
            fake_score = self.discriminator(gen_image)
            # real_output = tf.concat([tf.ones((self.batch_size*3, real_score.shape[1], real_score.shape[2], 1), dtype="float32"), tf.zeros((self.batch_size*3, real_score.shape[1], real_score.shape[2], 1), dtype="float32")], axis=-1)
            # fake_output = tf.concat([tf.zeros((self.batch_size*3, fake_score.shape[1], fake_score.shape[2], 1), dtype="float32"), tf.ones((self.batch_size*3, fake_score.shape[1], fake_score.shape[2], 1), dtype="float32")], axis=-1)
            # cce = tf.keras.losses.CategoricalCrossentropy()
            # d_loss = (cce(real_output, real_score) + cce(fake_output, fake_score)) * 0.5
            # real_gt = tf.ones_like(real_score)
            # fake_gt = tf.zeros_like(fake_score)
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
                for i in range(2):
                    d_loss = self.d_train_step(low_images, high_images, train=True)
                d_loss_batch.append(d_loss)

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
            self.encoder.save_weights('model_weight/patch_encoder_normal')
            self.ztozg.save_weights('model_weight/patch_ztozg_normal')
            self.generator.save_weights('model_weight/patch_g_normal')
            self.discriminator.save_weights('model_weight/patch_d_normal')
            # if epoch > 50:
            #   self.plot_image(epoch, self.test_path1, data_name='test_21ID')
            if epoch == 150:
                self.plot_image(epoch, self.test_path1, data_name='test_21ID_normal')
                self.plot_image(epoch, self.test_path2, data_name='train_90ID_normal')
                self.plot_image(epoch, self.test_path3, data_name='train_celeba_normal')

        plt.plot(image_loss_epoch)
        plt.savefig('result/PatchGAN/image_loss_normal')
        plt.close()

        plt.plot(style_loss_epoch)
        plt.savefig('result/PatchGAN/style_loss_normal')
        plt.close()

        plt.plot(g_loss_epoch)
        plt.savefig('result/PatchGAN/g_loss_normal')
        plt.close()

        plt.plot(d_loss_epoch)
        plt.savefig('result/PatchGAN/d_loss_normal')
        plt.close()

        plt.plot(g_loss_epoch)
        plt.plot(d_loss_epoch)
        plt.legend(['g_loss', 'd_loss'], loc='upper right')
        plt.savefig('result/PatchGAN/adv_loss_normal')
        plt.close()

    def plot_image(self, epoch, path, data_name):
        plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        for num, filename in enumerate(path):
            image = cv2.imread(filename, 0) / 255
            low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z1, z2, z3 = self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
            zg1, _, _ = self.ztozg(z1)
            zg2, _, _ = self.ztozg(z2)
            zg3, _, _ = self.ztozg(z3)

            gen1_image = self.generator(zg1)
            gen2_image = self.generator(zg2)
            gen3_image = self.generator(zg3)

            plt.subplot(7, 7, count + 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')

            plt.subplot(7, 7, count + 8)
            plt.axis('off')
            plt.imshow(low1_image, cmap='gray')

            plt.subplot(7, 7, count + 15)
            plt.axis('off')
            plt.imshow(tf.reshape(gen1_image, [64, 64]), cmap='gray')

            plt.subplot(7, 7, count + 22)
            plt.axis('off')
            plt.imshow(low2_image, cmap='gray')

            plt.subplot(7, 7, count + 29)
            plt.axis('off')
            plt.imshow(tf.reshape(gen2_image, [64, 64]), cmap='gray')

            plt.subplot(7, 7, count + 36)
            plt.axis('off')
            plt.imshow(low3_image, cmap='gray')

            plt.subplot(7, 7, count + 43)
            plt.axis('off')
            plt.imshow(tf.reshape(gen3_image, [64, 64]), cmap='gray')
            count += 1

            if (num+1) % 7 == 0:
                plt.savefig(f'result/PatchGAN/{data_name}_{epoch}_{num+1}image')
                plt.close()
                plt.subplots(figsize=(7, 7))
                plt.subplots_adjust(hspace=0, wspace=0)
                count = 0

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

class zd_zg_distillation():
    def __init__(self, epochs, batch_num, batch_size):
        # set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.opti = tf.keras.optimizers.Adam(1e-4)

        # set the model
        self.encoder = normal_encoder()
        self.ztozd = ZtoZd()
        self.ztozg = ZtoZg()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozd.load_weights('model_weight/AE_ztozd')
        self.ztozg.load_weights('model_weight/patch_ztozg')
        self.generator.load_weights('model_weight/patch_g')
        self.discriminator.load_weights('model_weight/patch_d')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        # set the data path
        self.train_path, self.train_label, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.train_label.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = "/home/bosen/PycharmProjects/SRGAN_learning_based_inversion/celeba_train/"
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
        path_AR_real_train = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_train90/"
        path_AR_real_test = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_test21/"
        train_path, train_label = [], []
        test_path1, test_path2, test_path3 = [], [], []
        ID = [f'ID{i}' for i in range(1, 91)]

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2070:
                train_path.append(path_celeba + filename)
                train_label.append(tf.one_hot(90, 91))
            if num < 21:
                test_path3.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                    train_label.append(tf.one_hot(int(id[2:])-1, 91))
                if num == 21:
                    test_path2.append(path_AR_syn_train + id + '/' + filename)

        for count, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_AR_real_train + id)):
                if '-1-0' in filename or '-1-1' in filename or '-1-2' in filename:
                    train_path.append(path_AR_real_train + id + '/' + filename)
                    train_label.append(tf.one_hot(int(id[2:])-1, 91))
                # if '-1-0' in filename and count < 21:
                #     test_path2.append(path_AR_real_train + id + '/' + filename)

        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if '11_test' in filename:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)

        train_path, train_label, test_path1, test_path2, test_path3 = np.array(train_path), np.array(train_label), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        train_data = list(zip(train_path, train_label))
        np.random.shuffle(train_data)
        train_data = list(zip(*train_data))
        return  np.array(train_data[0]), np.array(train_data[1]), test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size, image=True):
        high_images, low_images = [], []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        if image:
            zHs, zhs, zms, zls = [], [], [], []
            for path in train_data:
                image = cv2.imread(path, 0) / 255
                if "AR" in path:
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                zH = self.encoder(image.reshape(1, 64, 64, 1))
                zh = self.encoder(low1_image)
                zm = self.encoder(low2_image)
                zl = self.encoder(low3_image)

                zHs.append(tf.reshape(zH, [256]))
                zhs.append(tf.reshape(zh, [256]))
                zms.append(tf.reshape(zm, [256]))
                zls.append(tf.reshape(zl, [256]))

                low_images.append(low1_image), low_images.append(low2_image), low_images.append(low3_image)
                high_images.append(image), high_images.append(image), high_images.append(image)

            zHs, zhs, zms, zls = np.array(zHs), np.array(zhs), np.array(zms), np.array(zls)
            high_images = np.array(high_images).reshape(-1, 64, 64, 1)
            low_images = np.array(low_images).reshape(-1, 64, 64, 1)
            return high_images, low_images, zHs, zhs, zms, zls
        else:
            labels = []
            for label in train_data:
                for i in range(1):
                    labels.append(label)
            labels = np.array(labels)
            return labels

    def style_loss(self, real, fake):
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = self.feature_extraction(real)
        fake_feature = self.feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def distillation_loss(self, zd, zg):
        dot_product_d_space = tf.matmul(zd, tf.transpose(zd))
        dot_product_g_space = tf.matmul(zg, tf.transpose(zg))
        square_norm_d_space = tf.linalg.diag_part(dot_product_d_space)
        square_norm_g_space = tf.linalg.diag_part(dot_product_g_space)

        distances_d_space = tf.sqrt(tf.expand_dims(square_norm_d_space, 1) - 2.0 * dot_product_d_space + tf.expand_dims(square_norm_d_space,
                                                                                                0) + 1e-8)
        distances_g_space = tf.sqrt(tf.expand_dims(square_norm_g_space, 1) - 2.0 * dot_product_g_space + tf.expand_dims(square_norm_g_space,
                                                                                                0) + 1e-8)
        norm_distances_d_space = distances_d_space / (tf.reduce_sum(distances_d_space / 2))
        norm_distances_g_space = distances_g_space / (tf.reduce_sum(distances_g_space / 2))
        distance = tf.math.abs(norm_distances_d_space - norm_distances_g_space)
        distance = tf.reduce_sum(distance) / 2
        return distance

    def train_step(self, low_images, high_images, label, zHs, zhs, zms, zls, train='total'):
        cce = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            z = self.encoder(low_images)
            zg, _, _ = self.ztozg(z)
            gen_images = self.generator(zg)
            fake_score = self.discriminator(gen_images)

            zdH, _ = self.ztozd(zHs)
            zgh, _, predh = self.ztozg(zhs)
            zgm, _, predm = self.ztozg(zms)
            zgl, _, predl = self.ztozg(zls)

            dis1 = self.distillation_loss(zdH, zgh)
            dis2 = self.distillation_loss(zdH, zgm)
            dis3 = self.distillation_loss(zdH, zgl)

            ce1 = cce(label, predh)
            ce2 = cce(label, predm)
            ce3 = cce(label, predl)


            adv_loss = tf.reduce_mean(tf.square(fake_score - 1))
            image_loss = 100 * tf.reduce_mean(tf.square(high_images - gen_images))
            style_loss = 20 * self.style_loss(high_images, gen_images)
            dis_loss = dis1 + dis2 + dis3
            ce_loss = 0.1 * (ce1 + ce2 + ce3)

            total_loss = image_loss + style_loss + adv_loss + ce_loss + dis_loss
            ztozg_loss = ce_loss + dis_loss
            g_loss = image_loss + style_loss + adv_loss

        if train == "total":
            grads = tape.gradient(total_loss, self.generator.trainable_variables + self.ztozg.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.generator.trainable_variables + self.ztozg.trainable_variables))
        if train == 'ztozg':
            grads = tape.gradient(ztozg_loss, self.ztozg.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.ztozg.trainable_variables))
        if train == 'g':
            grads = tape.gradient(g_loss, self.generator.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.generator.trainable_variables))
        return image_loss, style_loss, adv_loss, dis_loss, ce_loss

    def training(self):
        image_loss_epoch = []
        style_loss_epoch = []
        adv_loss_epoch = []
        dis_loss_epoch = []
        ce_loss_epoch = []

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            adv_loss_batch = []
            dis_loss_batch = []
            ce_loss_batch = []

            if epoch < 30:
                train_item = 'ztozg'
            elif 50 > epoch >= 30:
                train_item = 'g'
            else:
                train_item = 'total'

            for batch in range(self.batch_num):
                high_images, low_images, zH, zh, zm, zl = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                label = self.get_batch_data(self.train_label, batch, batch_size=self.batch_size, image=False)

                image_loss, style_loss, adv_loss, dis_loss, ce_loss = self.train_step(low_images, high_images, label, zH, zh, zm, zl, train=train_item)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                adv_loss_batch.append(adv_loss)
                dis_loss_batch.append(dis_loss)
                ce_loss_batch.append(ce_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            adv_loss_epoch.append(np.mean(adv_loss_batch))
            dis_loss_epoch.append(np.mean(dis_loss_batch))
            ce_loss_epoch.append(np.mean(ce_loss_batch))

            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the adv_loss is {adv_loss_epoch[-1]}')
            print(f'the dis_loss is {dis_loss_epoch[-1]}')
            print(f'the ce_loss is {ce_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.ztozg.save_weights('model_weight/zd_zg_distillation_ztozg')
            self.generator.save_weights('model_weight/zd_zg_distillation_generator')
            self.plot_image(epoch, self.test_path1, data_name='test_21ID')
            if epoch == self.epochs:
                self.plot_image(epoch, self.test_path2, data_name='train_90ID')
                self.plot_image(epoch, self.test_path3, data_name='train_celeba')

        plt.plot(image_loss_epoch)
        plt.savefig('result/zd_zg_distillation/image_loss')
        plt.close()

        plt.plot(style_loss_epoch)
        plt.savefig('result/zd_zg_distillation/style_loss')
        plt.close()

        plt.plot(adv_loss_epoch)
        plt.savefig('result/zd_zg_distillation/adv_loss')
        plt.close()

        plt.plot(dis_loss_epoch)
        plt.savefig('result/zd_zg_distillation/dis_loss')
        plt.close()

        plt.plot(ce_loss_epoch)
        plt.savefig('result/zd_zg_distillation/ce_loss')
        plt.close()

    def plot_image(self, epoch, path, data_name):
        plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        for num, filename in enumerate(path):
            image = cv2.imread(filename, 0) / 255
            low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z1, z2, z3 = self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
            zg1, _, _ = self.ztozg(z1)
            zg2, _, _ = self.ztozg(z2)
            zg3, _, _ = self.ztozg(z3)

            gen1_image = self.generator(zg1)
            gen2_image = self.generator(zg2)
            gen3_image = self.generator(zg3)

            plt.subplot(7, 7, count + 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')

            plt.subplot(7, 7, count + 8)
            plt.axis('off')
            plt.imshow(low1_image, cmap='gray')

            plt.subplot(7, 7, count + 15)
            plt.axis('off')
            plt.imshow(tf.reshape(gen1_image, [64, 64]), cmap='gray')

            plt.subplot(7, 7, count + 22)
            plt.axis('off')
            plt.imshow(low2_image, cmap='gray')

            plt.subplot(7, 7, count + 29)
            plt.axis('off')
            plt.imshow(tf.reshape(gen2_image, [64, 64]), cmap='gray')

            plt.subplot(7, 7, count + 36)
            plt.axis('off')
            plt.imshow(low3_image, cmap='gray')

            plt.subplot(7, 7, count + 43)
            plt.axis('off')
            plt.imshow(tf.reshape(gen3_image, [64, 64]), cmap='gray')
            count += 1

            if (num+1) % 7 == 0:
                plt.savefig(f'result/zd_zg_distillation/{data_name}_{epoch}_{num+1}image')
                plt.close()
                plt.subplots(figsize=(7, 7))
                plt.subplots_adjust(hspace=0, wspace=0)
                count = 0

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
            plt.savefig(f'result/zd_zg_distillation/train_id{id_index}_similarity_syn')
            plt.close()
        else:
            plt.savefig(f'result/zd_zg_distillation/test_id{id_index}_similarity_syn')
            plt.close()

    def different_resolution_synthesis_result(self, train=True):
        plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        if train:
            path = self.test_path2
        else:
            path = self.test_path1
        for num, filename in enumerate(path):
            if num == 21:
                break
            image = cv2.imread(filename, 0) / 255
            low1_image = cv2.resize(cv2.resize(image, (24, 24), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (12, 12), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z1, z2, z3 = self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(
                low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
            zg1, _, _ = self.ztozg(z1)
            zg2, _, _ = self.ztozg(z2)
            zg3, _, _ = self.ztozg(z3)

            gen1_image = self.generator(zg1)
            gen2_image = self.generator(zg2)
            gen3_image = self.generator(zg3)

            plt.subplot(7, 7, count + 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')

            plt.subplot(7, 7, count + 8)
            plt.axis('off')
            plt.imshow(low1_image, cmap='gray')

            plt.subplot(7, 7, count + 15)
            plt.axis('off')
            plt.imshow(tf.reshape(gen1_image, [64, 64]), cmap='gray')

            plt.subplot(7, 7, count + 22)
            plt.axis('off')
            plt.imshow(low2_image, cmap='gray')

            plt.subplot(7, 7, count + 29)
            plt.axis('off')
            plt.imshow(tf.reshape(gen2_image, [64, 64]), cmap='gray')

            plt.subplot(7, 7, count + 36)
            plt.axis('off')
            plt.imshow(low3_image, cmap='gray')

            plt.subplot(7, 7, count + 43)
            plt.axis('off')
            plt.imshow(tf.reshape(gen3_image, [64, 64]), cmap='gray')
            count += 1

            if (num + 1) % 7 == 0:
                if train:
                    plt.savefig(f'result/zd_zg_distillation/train_different_reo_syn_result_{num + 1}image')
                    plt.close()
                    plt.subplots(figsize=(7, 7))
                    plt.subplots_adjust(hspace=0, wspace=0)
                    count = 0
                else:
                    plt.savefig(f'result/zd_zg_distillation/test_different_reo_syn_result_{num + 1}image')
                    plt.close()
                    plt.subplots(figsize=(7, 7))
                    plt.subplots_adjust(hspace=0, wspace=0)
                    count = 0

class Regression():
    def __init__(self, epochs, batch_num, batch_size):
        # set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        # set the model
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.with_instance_regression = regression_model_with_instance()
        self.without_instance_regression = regression_model_without_instance()
        self.regression = regression_model_with_instance()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')

        self.regression.load_weights('model_weight/regression_one_to_one')
        self.with_instance_regression.load_weights('model_weight/regression2')
        self.without_instance_regression.load_weights('model_weight/regression')

        self.generator.load_weights('model_weight/zd_zg_distillation_generator')
        self.discriminator.load_weights('model_weight/patch_d')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        # self.zgHs, self.zghs, self.zgms, self.zgls, self.zghs_intepolation, self.zgms_intepolation, self.zgls_intepolation, self.zgh_neg, self.zgm_neg, self.zgl_neg, self.train_path, self.label = self.AR_regression_training_data()

    def AR_regression_training_data(self):
        def get_all_feature():
            path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
            data_path, label = [[] for i in range(90)], []
            ID = [f'ID{i}' for i in range(1, 91)]

            for num, id in enumerate(ID):
                for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                    if 20 <= count < 43:
                        data_path[num].append(path_AR_syn_train + id + '/' + filename)
                        label.append(int(id[2:]))

            zgHs, zghs, zgms, zgls = [], [], [], []
            number_each_id = 0
            for num, id in enumerate(data_path):
                number_each_id = 0
                for count, path in enumerate(id):
                    number_each_id += 1
                    image = cv2.imread(path, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                    zH = self.encoder(image.reshape(1, 64, 64, 1))
                    zh = self.encoder(low1_image)
                    zm = self.encoder(low2_image)
                    zl = self.encoder(low3_image)
                    zgH, _, _ = self.ztozg(zH)
                    zgh, _, _ = self.ztozg(zh)
                    zgm, _, _ = self.ztozg(zm)
                    zgl, _, _ = self.ztozg(zl)

                    zgHs.append(tf.reshape(zgH, [200]))
                    zghs.append(tf.reshape(zgh, [200]))
                    zgms.append(tf.reshape(zgm, [200]))
                    zgls.append(tf.reshape(zgl, [200]))

            zgHs, zghs, zgms, zgls, label = np.array(zgHs), np.array(zghs), np.array(zgms), np.array(zgls), np.array(label)
            data_path = np.array(data_path)
            return zgHs, zghs, zgms, zgls, number_each_id, np.array(data_path).reshape(-1), label

        def calculate_cosine_similarity(zgs):
            def set_diagonal_blocks_to_zero(matrix):
                n = matrix.shape[0]
                block_size = n // 90
                modified_matrix = np.copy(matrix)

                for i in range(90):
                    start_row = i * block_size
                    end_row = (i + 1) * block_size
                    modified_matrix[start_row:end_row, start_row:end_row] = 0

                return modified_matrix

            similarity_matrix = tf.matmul(zgs, tf.transpose(zgs)) / tf.matmul(tf.norm(zgs, axis=1, keepdims=True), tf.norm(tf.transpose(zgs), axis=0, keepdims=True))
            similarity_matrix = set_diagonal_blocks_to_zero(similarity_matrix)
            return similarity_matrix

        def find_max_cosine_similarity(similarity_matrix, num_neighbors=3):
            num_images = similarity_matrix.shape[0]
            max_similarity_indices = []
            for i in range(num_images):
                similarity_row = similarity_matrix[i]
                sorted_indices = np.argsort(similarity_row)[-num_neighbors:][::-1]
                max_similarity_indices.append(sorted_indices)
            return max_similarity_indices

        def intepolation(zghs, zgms, zgls, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair):
            zghs_intepolation, zgms_intepolation, zgls_intepolation = [[] for i in range(zghs.shape[0])], [[] for i in range(zgms.shape[0])], [[] for i in range(zgls.shape[0])]
            zghs_expand, zgms_expand, zgls_expand = tf.tile(tf.reshape(zghs, [-1, 1, 200]), [1, 3, 1]), tf.tile(tf.reshape(zgms, [-1, 1, 200]), [1, 3, 1]), tf.tile(tf.reshape(zgls, [-1, 1, 200]), [1, 3, 1])

            zg_data = list(zip(zghs_expand, zgms_expand, zgls_expand, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair))
            for num, (zgh, zgm, zgl, zgh_neg, zgm_neg, zgl_neg) in enumerate(zg_data):
                zghs_intepolation[num].append(zgh * 0.7 + zgh_neg * 0.3)
                zgms_intepolation[num].append(zgm * 0.7 + zgm_neg * 0.3)
                zgls_intepolation[num].append(zgl * 0.7 + zgl_neg * 0.3)
            zghs_intepolation = np.array(zghs_intepolation).reshape(-1, 3, 200)
            zgms_intepolation = np.array(zgms_intepolation).reshape(-1, 3, 200)
            zgls_intepolation = np.array(zgls_intepolation).reshape(-1, 3, 200)
            return zghs_intepolation, zgms_intepolation, zgls_intepolation

        zgHs, zghs, zgms, zgls, number_each_id, train_path, label = get_all_feature()

        zghs_negative_pair, zgms_negative_pair, zgls_negative_pair = [[] for i in range(zghs.shape[0])], [[] for i in range(zgms.shape[0])], [[] for i in range(zgls.shape[0])]

        similarity_matrix_zghs = calculate_cosine_similarity(zghs)
        similarity_matrix_zgms = calculate_cosine_similarity(zgms)
        similarity_matrix_zgls = calculate_cosine_similarity(zgls)
        max_cosine_similarity_zgh_index = find_max_cosine_similarity(similarity_matrix_zghs)
        max_cosine_similarity_zgm_index = find_max_cosine_similarity(similarity_matrix_zgms)
        max_cosine_similarity_zgl_index = find_max_cosine_similarity(similarity_matrix_zgls)

        max_cosine_similarity_index = list(zip(max_cosine_similarity_zgh_index, max_cosine_similarity_zgm_index, max_cosine_similarity_zgl_index))
        for num, (zghs_index, zgms_index, zgls_index) in enumerate(max_cosine_similarity_index):
            for i in range(zghs_index.shape[0]):
                zghs_negative_pair[num].append(zghs[zghs_index[i]])
            for i in range(zgms_index.shape[0]):
                zgms_negative_pair[num].append(zgms[zgms_index[i]])
            for i in range(zgls_index.shape[0]):
                zgls_negative_pair[num].append(zgls[zgls_index[i]])
        zghs_negative_pair, zgms_negative_pair, zgls_negative_pair = np.array(zghs_negative_pair), np.array(zgms_negative_pair), np.array(zgls_negative_pair)
        zghs_intepolation, zgms_intepolation, zgls_intepolation = intepolation(zghs, zgms, zgls, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair)

        # for i in [1, 50, 75, 2000, 2060]:
        #     plt.subplots(figsize=(4, 2))
        #     plt.subplots_adjust(hspace=0, wspace=0)
        #
        #     plt.subplot(2, 4, 1)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls[i], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 2)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_negative_pair[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 3)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_negative_pair[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 4)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_negative_pair[i][2], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 5)
        #     plt.axis('off')
        #     plt.imshow(np.zeros((64, 64)), cmap='gray')
        #     plt.subplot(2, 4, 6)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_intepolation[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 7)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 8)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_intepolation[i][2], [1, 200])), [64, 64]), cmap='gray')
        #     plt.show()

        return zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair, train_path, label

    def Celeba_regression_training_data(self):
        def get_all_feature():
            path_celeba = "/home/bosen/PycharmProjects/SRGAN_learning_based_inversion/celeba_train/"
            train_path = []
            train_label = []

            for count, filename in enumerate(os.listdir(path_celeba)):
                if count == 1000:
                    break
                train_path.append(path_celeba + filename)
                train_label.append(tf.one_hot(90, 91))

            zgHs, zghs, zgms, zgls = [], [], [], []
            for count, path in enumerate(train_path):
                image = cv2.imread(path, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                zH = self.encoder(image.reshape(1, 64, 64, 1))
                zh = self.encoder(low1_image)
                zm = self.encoder(low2_image)
                zl = self.encoder(low3_image)
                zgH, _, _ = self.ztozg(zH)
                zgh, _, _ = self.ztozg(zh)
                zgm, _, _ = self.ztozg(zm)
                zgl, _, _ = self.ztozg(zl)

                zgHs.append(tf.reshape(zgH, [200]))
                zghs.append(tf.reshape(zgh, [200]))
                zgms.append(tf.reshape(zgm, [200]))
                zgls.append(tf.reshape(zgl, [200]))

            zgHs, zghs, zgms, zgls = np.array(zgHs), np.array(zghs), np.array(zgms), np.array(zgls)
            return zgHs, zghs, zgms, zgls, np.array(train_path), np.array(train_label)

        def calculate_cosine_similarity(zgs):
            similarity_matrix = tf.matmul(zgs, tf.transpose(zgs)) / tf.matmul(tf.norm(zgs, axis=1, keepdims=True), tf.norm(tf.transpose(zgs), axis=0, keepdims=True))
            similarity_matrix_diagnonal = tf.linalg.diag_part(similarity_matrix+1)
            similarity_matrix -= tf.linalg.diag(similarity_matrix_diagnonal)
            return similarity_matrix

        def find_max_cosine_similarity(similarity_matrix, num_neighbors=3):
            num_images = similarity_matrix.shape[0]
            max_similarity_indices = []
            for i in range(num_images):
                similarity_row = similarity_matrix[i]
                sorted_indices = np.argsort(similarity_row)[-num_neighbors:][::-1]
                max_similarity_indices.append(sorted_indices)
            return max_similarity_indices

        def intepolation(zghs, zgms, zgls, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair):
            zghs_intepolation, zgms_intepolation, zgls_intepolation = [[] for i in range(zghs.shape[0])], [[] for i in range(zgms.shape[0])], [[] for i in range(zgls.shape[0])]
            zghs_expand, zgms_expand, zgls_expand = tf.tile(tf.reshape(zghs, [-1, 1, 200]), [1, 3, 1]), tf.tile(tf.reshape(zgms, [-1, 1, 200]), [1, 3, 1]),\
                                                    tf.tile(tf.reshape(zgls, [-1, 1, 200]), [1, 3, 1])

            zg_data = list(zip(zghs_expand, zgms_expand, zgls_expand, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair))
            for num, (zgh, zgm, zgl, zgh_neg, zgm_neg, zgl_neg) in enumerate(zg_data):
                zghs_intepolation[num].append(zgh*0.7 + zgh_neg*0.3)
                zgms_intepolation[num].append(zgm*0.7 + zgm_neg*0.3)
                zgls_intepolation[num].append(zgl*0.7 + zgl_neg*0.3)
            zghs_intepolation = np.array(zghs_intepolation).reshape(-1, 3, 200)
            zgms_intepolation = np.array(zgms_intepolation).reshape(-1, 3, 200)
            zgls_intepolation = np.array(zgls_intepolation).reshape(-1, 3, 200)
            return zghs_intepolation, zgms_intepolation, zgls_intepolation

        zgHs, zghs, zgms, zgls, train_path, train_label = get_all_feature()
        zghs_negative_pair, zgms_negative_pair, zgls_negative_pair = [[] for i in range(zghs.shape[0])], [[] for i in range(zgms.shape[0])], [[] for i in range(zgls.shape[0])]

        similarity_matrix_zghs = calculate_cosine_similarity(zghs)
        similarity_matrix_zgms = calculate_cosine_similarity(zgms)
        similarity_matrix_zgls = calculate_cosine_similarity(zgls)
        max_cosine_similarity_zgh_index = find_max_cosine_similarity(similarity_matrix_zghs)
        max_cosine_similarity_zgm_index = find_max_cosine_similarity(similarity_matrix_zgms)
        max_cosine_similarity_zgl_index = find_max_cosine_similarity(similarity_matrix_zgls)

        max_cosine_similarity_index = list(zip(max_cosine_similarity_zgh_index, max_cosine_similarity_zgm_index, max_cosine_similarity_zgl_index))
        for num, (zghs_index, zgms_index, zgls_index) in enumerate(max_cosine_similarity_index):
            for i in range(zghs_index.shape[0]):
                zghs_negative_pair[num].append(zghs[zghs_index[i]])
            for i in range(zgms_index.shape[0]):
                zgms_negative_pair[num].append(zgms[zgms_index[i]])
            for i in range(zgls_index.shape[0]):
                zgls_negative_pair[num].append(zgls[zgls_index[i]])
        zghs_negative_pair, zgms_negative_pair, zgls_negative_pair = np.array(zghs_negative_pair), np.array(zgms_negative_pair), np.array(zgls_negative_pair)
        zghs_intepolation, zgms_intepolation, zgls_intepolation = intepolation(zghs, zgms, zgls, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair)

        # for i in range(5):
        #     plt.subplots(figsize=(4, 2))
        #     plt.subplots_adjust(hspace=0, wspace=0)
        #
        #     plt.subplot(2, 4, 1)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls[i], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 2)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_negative_pair[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 3)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_negative_pair[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 4)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_negative_pair[i][2], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 5)
        #     plt.axis('off')
        #     plt.imshow(np.zeros((64, 64)), cmap='gray')
        #     plt.subplot(2, 4, 6)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_intepolation[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 7)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(2, 4, 8)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgls_intepolation[i][2], [1, 200])), [64, 64]), cmap='gray')
        #     plt.show()
        return zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair, train_path

    def reg_loss(self, zgH, zgh, zgm, zgl, zgh_intepolation, zgm_intepolation, zgl_intepolation, label):
        reg_loss = 0
        zregHs = self.regression(zgH)
        zreghs = self.regression(zgh)
        zregms = self.regression(zgm)
        zregls = self.regression(zgl)

        zregh_intepolation = self.regression(zgh_intepolation.reshape(-1, 200))
        zregm_intepolation = self.regression(zgm_intepolation.reshape(-1, 200))
        zregl_intepolation = self.regression(zgl_intepolation.reshape(-1, 200))
        # label_intepolation = tf.cast([label[i] for i in range(label.shape[0]) for x in range(3)], dtype=tf.float32)
        #
        #
        # number_of_error = 0
        # for num, (zregH, zregh, zregm, zregl) in enumerate(zip(zregHs, zreghs, zregms, zregls)):
        #     indices = np.where(self.label == label[num])[0]
        #     corresponding_zgHs_value = tf.cast([self.zgHs[i] for i in indices], dtype=tf.float32)
        #
        #     zregH_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregH, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     zregh_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregh, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     zregm_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregm, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     zregl_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregl, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     reg_loss += (zregH_error + zregh_error + zregm_error + zregl_error)
        #     # reg_loss += zregh_error + zregm_error + zregl_error
        #     number_of_error += 4
        #
        # for num, (zregh_aug, zregm_aug, zregl_aug) in enumerate(zip(zregh_intepolation, zregm_intepolation, zregl_intepolation)):
        #     indices = np.where(self.label == label_intepolation[num])[0]
        #     corresponding_zgHs_value = tf.cast([self.zgHs[i] for i in indices], dtype=tf.float32)
        #     zregh_aug_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregh_aug, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     zregm_aug_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregm_aug, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     zregl_aug_error = tf.reduce_min(tf.reduce_mean(tf.square(tf.tile(tf.reshape(zregl_aug, [1, 200]), [corresponding_zgHs_value.shape[0], 1]) - corresponding_zgHs_value), axis=-1))
        #     reg_loss += (zregh_aug_error + zregm_aug_error + zregl_aug_error)
        #     number_of_error += 3

        zgH_inte = tf.reshape(tf.tile(zgH.reshape(-1, 1, 200), [1, 3, 1]), [-1, 200])
        reg_loss = tf.reduce_mean(tf.square(zgH - zregHs)) + tf.reduce_mean(tf.square(zgH - zreghs)) + tf.reduce_mean(tf.square(zgH - zregms)) + \
                   tf.reduce_mean(tf.square(zgH - zregls)) + tf.reduce_mean(tf.square(zgH_inte - zregh_intepolation)) + tf.reduce_mean(tf.square(zgH_inte - zregm_intepolation)) + \
                   tf.reduce_mean(tf.square(zgH_inte - zregl_intepolation))
        return reg_loss / 6

    def image_loss(self, zgH, zgh, zgm, zgl, zgh_intepolation, zgm_intepolation, zgl_intepolation, high_reso_images):
        # plt.subplots(figsize=(10, 8))
        # plt.subplots_adjust(hspace=0, wspace=0)
        # for i in range(10):
        #     plt.subplot(14, 10, i + 1)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(high_reso_images[i], [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 11)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgH[i], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 21)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgh[i], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 31)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgm[i], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 41)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgl[i], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 51)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgh_intepolation[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 61)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgh_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 71)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgh_intepolation[i][2], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 81)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgm_intepolation[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 91)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgm_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 101)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgm_intepolation[i][2], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 111)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgl_intepolation[i][0], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 121)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgl_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
        #     plt.subplot(14, 10, i + 131)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(self.generator(tf.reshape(zgl_intepolation[i][2], [1, 200])), [64, 64]), cmap='gray')
        # plt.show()
        zregH = self.regression(zgH)
        zregh = self.regression(zgh)
        zregm = self.regression(zgm)
        zregl = self.regression(zgl)

        zregh_inte = self.regression(zgh_intepolation.reshape(-1, 200))
        zregm_inte = self.regression(zgm_intepolation.reshape(-1, 200))
        zregl_inte = self.regression(zgl_intepolation.reshape(-1, 200))

        high_inte = tf.cast(tf.reshape(tf.tile(high_reso_images.reshape(-1, 1, 64, 64, 1), [1, 3, 1, 1, 1]), [-1, 64, 64, 1]), dtype=tf.float32)
        syn_H, syn_h, syn_m, syn_l = self.generator(zregH), self.generator(zregh), self.generator(zregm), self.generator(zregl)
        syn_h_inte, syn_m_inte, syn_l_inte = self.generator(zregh_inte), self.generator(zregm_inte), self.generator(zregl_inte)
        # print(high_inte.shape, syn_h_inte.shape, syn_m_inte.shape, syn_l_inte.shape)
        # plt.subplots(figsize=(10, 5))
        # plt.subplots_adjust(hspace=0, wspace=0)
        # for i in range(10):
        #     plt.subplot(5, 10, i+1)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(high_inte[i], [64, 64]), cmap='gray')
        #     plt.subplot(5, 10, i+11)
        #     plt.axis('off')
        #     # plt.imshow(tf.reshape(syn_H[i], [64, 64]), cmap='gray')
        #     # plt.subplot(8, 10, i+21)
        #     # plt.axis('off')
        #     # plt.imshow(tf.reshape(syn_h[i], [64, 64]), cmap='gray')
        #     # plt.subplot(8, 10, i + 31)
        #     # plt.axis('off')
        #     # plt.imshow(tf.reshape(syn_m[i], [64, 64]), cmap='gray')
        #     # plt.subplot(8, 10, i + 41)
        #     # plt.axis('off')
        #     # plt.imshow(tf.reshape(syn_h[i], [64, 64]), cmap='gray')
        #     plt.subplot(5, 10, i + 21)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_h_inte[i], [64, 64]), cmap='gray')
        #     plt.subplot(5, 10, i + 31)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_m_inte[i], [64, 64]), cmap='gray')
        #     plt.subplot(5, 10, i + 41)
        #     plt.axis('off')
        #     plt.imshow(tf.reshape(syn_l_inte[i], [64, 64]), cmap='gray')
        # plt.show()
        image_loss = tf.reduce_mean(tf.square(high_reso_images.reshape(-1, 64, 64, 1) - syn_H)) + \
                     tf.reduce_mean(tf.square(high_reso_images.reshape(-1, 64, 64, 1) - syn_h)) + \
                     tf.reduce_mean(tf.square(high_reso_images.reshape(-1, 64, 64, 1) - syn_m)) + \
                     tf.reduce_mean(tf.square(high_reso_images.reshape(-1, 64, 64, 1) - syn_l)) + \
                     tf.reduce_mean(tf.square(high_inte - syn_h_inte)) + \
                     tf.reduce_mean(tf.square(high_inte - syn_m_inte)) + \
                     tf.reduce_mean(tf.square(high_inte - syn_l_inte))
        return image_loss / 6

    def style_loss(self, zgH, zgh, zgm, zgl, zgh_intepolation, zgm_intepolation, zgl_intepolation, high_reso_images):

        def style_loss_subfunction(real, fake):
            real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
            real = tf.image.grayscale_to_rgb(real)
            fake = tf.image.grayscale_to_rgb(fake)

            real_feature = self.feature_extraction(real)
            fake_feature = self.feature_extraction(fake)
            distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
            return distance

        zregH = self.regression(zgH)
        zregh = self.regression(zgh)
        zregm = self.regression(zgm)
        zregl = self.regression(zgl)

        zregh_inte = self.regression(zgh_intepolation.reshape(-1, 200))
        zregm_inte = self.regression(zgm_intepolation.reshape(-1, 200))
        zregl_inte = self.regression(zgl_intepolation.reshape(-1, 200))
        high_inte = tf.cast(tf.reshape(tf.tile(high_reso_images.reshape(-1, 1, 64, 64, 1), [1, 3, 1, 1, 1]), [-1, 64, 64, 1]),dtype=tf.float32)
        syn_H, syn_h, syn_m, syn_l = self.generator(zregH), self.generator(zregh), self.generator(zregm), self.generator(zregl)
        syn_h_inte, syn_m_inte, syn_l_inte = self.generator(zregh_inte), self.generator(zregm_inte), self.generator(zregl_inte)
        style_loss = style_loss_subfunction(high_reso_images, syn_H) + \
                     style_loss_subfunction(high_reso_images, syn_h) + \
                     style_loss_subfunction(high_reso_images, syn_m) + \
                     style_loss_subfunction(high_reso_images, syn_l) + \
                     style_loss_subfunction(high_inte, syn_h_inte) + \
                     style_loss_subfunction(high_inte, syn_m_inte) + \
                     style_loss_subfunction(high_inte, syn_l_inte)
        return style_loss / 6

    def adv_loss(self, zgH, zgh, zgm, zgl, zgh_intepolation, zgm_intepolation, zgl_intepolation):
        zregH = self.regression(zgH)
        zregh = self.regression(zgh)
        zregm = self.regression(zgm)
        zregl = self.regression(zgl)

        zregh_inte = self.regression(zgh_intepolation.reshape(-1, 200))
        zregm_inte = self.regression(zgm_intepolation.reshape(-1, 200))
        zregl_inte = self.regression(zgl_intepolation.reshape(-1, 200))

        syn_H, syn_h, syn_m, syn_l = self.generator(zregH), self.generator(zregh), self.generator(zregm), self.generator(zregl)
        syn_h_inte, syn_m_inte, syn_l_inte = self.generator(zregh_inte), self.generator(zregm_inte), self.generator(zregl_inte)
        syn_H_score, syn_h_score, syn_m_score, syn_l_score = self.discriminator(syn_H), self.discriminator(syn_h), self.discriminator(syn_m), self.discriminator(syn_l)
        syn_h_inte_score, syn_m_inte_score, syn_l_inte_score = self.discriminator(syn_h_inte), self.discriminator(syn_m_inte), self.discriminator(syn_l_inte)

        adv_loss = tf.reduce_mean(tf.square(syn_H_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_h_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_m_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_l_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_h_inte_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_m_inte_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_l_inte_score - 1))
        return adv_loss / 6

    def constrast_loss(self, zgH, zgh_pos, zgm_pos, zgl_pos, zgh_neg, zgm_neg, zgl_neg):
        zregh_pos = self.regression(zgh_pos)
        zregm_pos = self.regression(zgm_pos)
        zregl_pos = self.regression(zgl_pos)

        zregh_neg, zregm_neg, zregl_neg = [], [], []
        zg_neg = list(zip(zgh_neg, zgm_neg, zgl_neg))
        for (zgh_n, zgm_n, zgl_n) in zg_neg:
            zregh_neg.append(self.regression(zgh_n))
            zregm_neg.append(self.regression(zgm_n))
            zregl_neg.append(self.regression(zgl_n))
        zregh_neg, zregm_neg, zrel_neg = tf.cast(zregh_neg, dtype=tf.float32), tf.cast(zregm_neg, dtype=tf.float32), tf.cast(zregl_neg, dtype=tf.float32)

        #normalize unit vector
        zgH = zgH / tf.norm(zgH, axis=1, keepdims=True)
        zregh_pos = zregh_pos / tf.norm(zregh_pos, axis=1, keepdims=True)
        zregm_pos = zregm_pos / tf.norm(zregm_pos, axis=1, keepdims=True)
        zregl_pos = zregl_pos / tf.norm(zregl_pos, axis=1, keepdims=True)
        zregh_neg = zregh_neg / tf.norm(zregh_neg, axis=2, keepdims=True)
        zregm_neg = zregm_neg / tf.norm(zregm_neg, axis=2, keepdims=True)
        zregl_neg = zregl_neg / tf.norm(zregl_neg, axis=2, keepdims=True)


        zgH_x_zregh_pos = tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregh_pos)))))
        zgH_x_zregm_pos = tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregm_pos)))))
        zgH_x_zregl_pos = tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregl_pos)))))

        zgH_x_zregh_neg = tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregh_neg[:, 0, :]))))) + \
                            tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregh_neg[:, 1, :]))))) + \
                            tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregh_neg[:, 2, :])))))

        zgH_x_zregm_neg = tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregm_neg[:, 0, :]))))) + \
                            tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregm_neg[:, 1, :]))))) + \
                            tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregm_neg[:, 2, :])))))

        zgH_x_zregl_neg = tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregl_neg[:, 0, :]))))) + \
                            tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregl_neg[:, 1, :]))))) + \
                            tf.reduce_mean(tf.linalg.diag_part(tf.math.exp(tf.matmul(zgH, tf.transpose(zregl_neg[:, 2, :])))))

        positive_score = (zgH_x_zregh_pos + zgH_x_zregm_pos + zgH_x_zregl_pos)/3
        negative_score = (zgH_x_zregh_neg/3 + zgH_x_zregm_neg/3 + zgH_x_zregl_neg/3) / 3
        constrast_loss = -tf.math.log(positive_score / (positive_score + negative_score))
        return constrast_loss

    def get_batch_data(self, data, batch_idx, batch_size, image=False):
        high_images = []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        if image:
            for path in train_data:
                image = cv2.imread(path, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                high_images.append(image)

            high_images = np.array(high_images).reshape(-1, 64, 64, 1)
            return high_images
        else:
            return np.array(train_data)

    def train_step(self, high_images, label, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zgh_neg, zgm_neg, zgl_neg, opti, train='total'):
        with tf.GradientTape(persistent=True) as tape:
            image_loss = 15 * self.image_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, high_images)
            style_loss = 10 * self.style_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, high_images)
            reg_loss = 0.3 * self.reg_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, label)
            adv_loss = 50 * self.adv_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation)
            constrast_loss = 0.2 * self.constrast_loss(zgHs, zghs, zgms, zgls, zgh_neg, zgm_neg, zgl_neg)

            pre_train_loss = reg_loss + image_loss + style_loss + adv_loss
            stage1_loss = reg_loss + image_loss + style_loss + adv_loss
            total_loss = image_loss + style_loss + adv_loss + reg_loss + constrast_loss

        if train == "pre_train":
            grads = tape.gradient(pre_train_loss, self.regression.trainable_variables)
            opti.apply_gradients(zip(grads, self.regression.trainable_variables))
        elif train == 'train_stage1':
            grads = tape.gradient(stage1_loss, self.regression.trainable_variables)
            opti.apply_gradients(zip(grads, self.regression.trainable_variables))
        else:
            grads = tape.gradient(total_loss, self.regression.trainable_variables)
            opti.apply_gradients(zip(grads, self.regression.trainable_variables))

        return image_loss, style_loss, reg_loss, adv_loss, constrast_loss

    def plot_image_AR(self, epoch, train=True):
        if train:
            path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        else:
            path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
        forward_psnr, forward_ssim = [[], [], []], [[], [], []]
        reg_psnr, reg_ssim = [[], [], []], [[], [], []]

        plt.subplots(figsize=(7, 10))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        for num_id, ID in enumerate(os.listdir(path_AR_syn_test)):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if train:
                    name = '11_train'
                else:
                    name = '12_test'

                if name in filename:
                    image = cv2.imread(path_AR_syn_test + ID + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    zh = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    zm = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    zl = self.encoder(low3_image.reshape(1, 64, 64, 1))

                    zgh, _, _ = self.ztozg(zh)
                    zgm, _, _ = self.ztozg(zm)
                    zgl, _, _ = self.ztozg(zl)

                    zregh = self.regression(zgh)
                    zregm = self.regression(zgm)
                    zregl = self.regression(zgl)

                    forward_zh_syn, forward_zm_syn, forward_zl_syn = self.generator(zgh), self.generator(zgm), self.generator(zgl)
                    reg_zh_syn, reg_zm_syn, reg_zl_syn = self.generator(zregh), self.generator(zregm), self.generator(zregl)

                    forward_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                    forward_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                    forward_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                    reg_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                    reg_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                    reg_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
                    forward_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                    forward_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                    forward_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                    reg_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                    reg_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                    reg_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])


                    plt.subplot(10, 7, count + 1)
                    plt.axis('off')
                    plt.imshow(image, cmap='gray')
                    plt.subplot(10, 7, count + 8)
                    plt.axis('off')
                    plt.imshow(low1_image, cmap='gray')
                    plt.subplot(10, 7, count + 15)
                    plt.axis('off')
                    plt.imshow(low2_image, cmap='gray')
                    plt.subplot(10, 7, count + 22)
                    plt.axis('off')
                    plt.imshow(low3_image, cmap='gray')
                    plt.subplot(10, 7, count + 29)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zh_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 36)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zm_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 43)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zl_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 50)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zh_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 57)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zm_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 64)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zl_syn, [64, 64]), cmap='gray')
                    count += 1
                    if (count) % 7 == 0:
                        if train:
                            plt.savefig(f'result/regression/AR_train/train_{epoch}_{num_id}result')
                            plt.close()
                        else:
                            plt.savefig(f'result/regression/AR_train/test_{epoch}_{num_id}result')
                            plt.close()
                        plt.subplots(figsize=(7, 10))
                        plt.subplots_adjust(hspace=0, wspace=0)
                        count = 0
        plt.close()
        forward_psnr, forward_ssim, reg_psnr, reg_ssim = np.array(forward_psnr), np.array(forward_ssim), np.array(reg_psnr), np.array(reg_ssim)
        print(f'the mean psnr forward is {np.mean(forward_psnr)}')
        print(f'the mean psnr reg is {np.mean(reg_psnr)}')
        print(f'the mean ssim forward is {np.mean(forward_ssim)}')
        print(f'the mean ssim reg is {np.mean(reg_ssim)}')
        print('--------------------------------')
        if np.mean(forward_psnr) < np.mean(reg_psnr):
            self.regression.save_weights('model_weight/best_regression')

        total_psnr = list(zip(forward_psnr, reg_psnr))
        total_ssim = list(zip(forward_ssim, reg_ssim))

        for ratio, (forward, reg) in enumerate(total_psnr):
            forward = [round(num, 1) for num in forward]
            reg = [round(num, 1) for num in reg]

            x1 = np.arange(len(forward))
            x2 = np.arange(len(reg)) - 0.2

            # plt.title(f'mPSNR={np.mean(reg)}')
            plt.bar(x1, forward, label='List 1', alpha=0.5)
            plt.bar(x2, reg, label='List 2', alpha=0.5)

            for i, v in enumerate(forward):
                plt.text(i, v, str(v), ha='center', va='bottom')
            for i, v in enumerate(reg):
                plt.text(i, v, str(v), ha='center', va='bottom')

            plt.xticks(x1, x1 + 1)
            # plt.legend(['forward', 'regression'], loc='upper right')
            if train:
                plt.savefig(f'result/regression/AR_train/train_{epoch}_forward_vs_reg_{2**(ratio+1)}_ratio_PSNR')
                plt.close()
            else:
                plt.savefig(f'result/regression/AR_train/test_{epoch}_forward_vs_reg_{2 ** (ratio + 1)}_ratio_PSNR')
                plt.close()

        for ratio, (forward, reg) in enumerate(total_ssim):
            forward = [round(num, 2) for num in forward]
            reg = [round(num, 2) for num in reg]

            x1 = np.arange(len(forward))
            x2 = np.arange(len(reg)) - 0.2

            # plt.title(f'mSSIM={np.mean(reg)}')
            plt.bar(x1, forward, label='List 1', alpha=0.5)
            plt.bar(x2, reg, label='List 2', alpha=0.5)

            for i, v in enumerate(forward):
                plt.text(i, v, str(v), ha='center', va='bottom')
            for i, v in enumerate(reg):
                plt.text(i, v, str(v), ha='center', va='bottom')

            plt.xticks(x1, x1 + 1)
            # plt.legend(['forward', 'regression'], loc='upper right')
            if train:
                plt.savefig(f'result/regression/AR_train/train_{epoch}_forward_vs_reg_{2 ** (ratio + 1)}_ratio_SSIM')
                plt.close()
            else:
                plt.savefig(f'result/regression/AR_train/test_{epoch}_forward_vs_reg_{2 ** (ratio + 1)}_ratio_SSIM')
                plt.close()

    def compare_multiple_method(self):
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
        forward_psnr, forward_ssim = [[], [], []], [[], [], []]
        reg_ins_psnr, reg_ins_ssim = [[], [], []], [[], [], []]
        reg_psnr, reg_ssim = [[], [], []], [[], [], []]

        plt.subplots(figsize=(7, 13))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        for num_id, ID in enumerate(os.listdir(path_AR_syn_test)):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                name = '11_test'
                if name in filename:
                    image = cv2.imread(path_AR_syn_test + ID + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    zh = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    zm = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    zl = self.encoder(low3_image.reshape(1, 64, 64, 1))

                    zgh, _, _ = self.ztozg(zh)
                    zgm, _, _ = self.ztozg(zm)
                    zgl, _, _ = self.ztozg(zl)

                    zreghins = self.with_instance_regression(zgh)
                    zregmins = self.with_instance_regression(zgm)
                    zreglins = self.with_instance_regression(zgl)

                    zregh = self.without_instance_regression(zgh)
                    zregm = self.without_instance_regression(zgm)
                    zregl = self.without_instance_regression(zgl)

                    forward_zh_syn, forward_zm_syn, forward_zl_syn = self.generator(zgh), self.generator(zgm), self.generator(zgl)
                    reg_zh_syn, reg_zm_syn, reg_zl_syn = self.generator(zregh), self.generator(zregm), self.generator(zregl)
                    reg_zh_syn_ins, reg_zm_syn_ins, reg_zl_syn_ins = self.generator(zreghins), self.generator(zregmins), self.generator(zreglins)


                    forward_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                    forward_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                    forward_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                    reg_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                    reg_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                    reg_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
                    reg_ins_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn_ins, max_val=1)[0])
                    reg_ins_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn_ins, max_val=1)[0])
                    reg_ins_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn_ins, max_val=1)[0])

                    forward_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                    forward_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                    forward_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                    reg_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                    reg_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                    reg_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
                    reg_ins_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn_ins, max_val=1)[0])
                    reg_ins_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn_ins, max_val=1)[0])
                    reg_ins_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn_ins, max_val=1)[0])


                    plt.subplot(13, 7, count + 1)
                    plt.axis('off')
                    plt.imshow(image, cmap='gray')
                    plt.subplot(13, 7, count + 8)
                    plt.axis('off')
                    plt.imshow(low1_image, cmap='gray')
                    plt.subplot(13, 7, count + 15)
                    plt.axis('off')
                    plt.imshow(low2_image, cmap='gray')
                    plt.subplot(13, 7, count + 22)
                    plt.axis('off')
                    plt.imshow(low3_image, cmap='gray')
                    plt.subplot(13, 7, count + 29)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zh_syn, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 36)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zm_syn, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 43)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zl_syn, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 50)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zh_syn, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 57)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zm_syn, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 64)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zl_syn, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 71)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zh_syn_ins, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 78)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zm_syn_ins, [64, 64]), cmap='gray')
                    plt.subplot(13, 7, count + 85)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zl_syn_ins, [64, 64]), cmap='gray')
                    count += 1
                    if (count) % 7 == 0:
                        plt.savefig(f'result/regression/AR_train/compare_with_and_without_instance_quality_result')
                        plt.close()
                        plt.subplots(figsize=(7, 13))
                        plt.subplots_adjust(hspace=0, wspace=0)
                        count = 0
        plt.close()
        forward_psnr, forward_ssim, reg_psnr, reg_ssim, reg_ins_psnr, reg_ins_ssim = np.array(forward_psnr), np.array(forward_ssim), np.array(reg_psnr), np.array(reg_ssim), np.array(reg_ins_psnr), np.array(reg_ins_ssim)
        print(f'the mean psnr forward is {np.mean(forward_psnr)}')
        print(f'the mean psnr reg is {np.mean(reg_psnr)}')
        print(f'the mean psnr reg instance is {np.mean(reg_ins_psnr)}')
        print(f'the mean ssim forward is {np.mean(forward_ssim)}')
        print(f'the mean ssim reg is {np.mean(reg_ssim)}')
        print(f'the mean ssim reg instance is {np.mean(reg_ins_ssim)}')
        print('--------------------------------')

        for i in range(3):
            plt.subplots(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(forward_psnr[i])
            plt.plot(reg_psnr[i])
            plt.plot(reg_ins_psnr[i])
            plt.legend(['forward', 'reg', 'reg_with_ins'], loc='upper right')

            plt.subplot(1, 2, 2)
            plt.plot(forward_ssim[i])
            plt.plot(reg_ssim[i])
            plt.plot(reg_ins_ssim[i])
            plt.legend(['forward', 'reg', 'reg_with_ins'], loc='upper right')
            plt.savefig(f'result/regression/AR_train/compare_with_and_without_instance_quantity_{2**(i+1)}_ratio_result')
            plt.close()

    def plot_image_celeba(self, epoch):
        path_celeba = "/home/bosen/PycharmProjects/SRGAN_learning_based_inversion/celeba_train/"
        forward_psnr, forward_ssim = [[], [], []], [[], [], []]
        reg_psnr, reg_ssim = [[], [], []], [[], [], []]

        plt.subplots(figsize=(7, 10))
        plt.subplots_adjust(hspace=0, wspace=0)
        num = 0
        for count, filename in enumerate(os.listdir(path_celeba)):
            if 3000 <= count < 3021:
                image = cv2.imread(path_celeba + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                zh = self.encoder(low1_image.reshape(1, 64, 64, 1))
                zm = self.encoder(low2_image.reshape(1, 64, 64, 1))
                zl = self.encoder(low3_image.reshape(1, 64, 64, 1))

                zgh, _, _ = self.ztozg(zh)
                zgm, _, _ = self.ztozg(zm)
                zgl, _, _ = self.ztozg(zl)

                zregh = self.regression(zgh)
                zregm = self.regression(zgm)
                zregl = self.regression(zgl)

                forward_zh_syn, forward_zm_syn, forward_zl_syn = self.generator(zgh), self.generator(zgm), self.generator(zgl)
                reg_zh_syn, reg_zm_syn, reg_zl_syn = self.generator(zregh), self.generator(zregm), self.generator(zregl)

                forward_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                forward_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                forward_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                reg_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                reg_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                reg_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
                forward_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                forward_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                forward_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                reg_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                reg_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                reg_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])


                plt.subplot(10, 7, num + 1)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
                plt.subplot(10, 7, num + 8)
                plt.axis('off')
                plt.imshow(low1_image, cmap='gray')
                plt.subplot(10, 7, num + 15)
                plt.axis('off')
                plt.imshow(low2_image, cmap='gray')
                plt.subplot(10, 7, num + 22)
                plt.axis('off')
                plt.imshow(low3_image, cmap='gray')
                plt.subplot(10, 7, num + 29)
                plt.axis('off')
                plt.imshow(tf.reshape(forward_zh_syn, [64, 64]), cmap='gray')
                plt.subplot(10, 7, num + 36)
                plt.axis('off')
                plt.imshow(tf.reshape(forward_zm_syn, [64, 64]), cmap='gray')
                plt.subplot(10, 7, num + 43)
                plt.axis('off')
                plt.imshow(tf.reshape(forward_zl_syn, [64, 64]), cmap='gray')
                plt.subplot(10, 7, num + 50)
                plt.axis('off')
                plt.imshow(tf.reshape(reg_zh_syn, [64, 64]), cmap='gray')
                plt.subplot(10, 7, num + 57)
                plt.axis('off')
                plt.imshow(tf.reshape(reg_zm_syn, [64, 64]), cmap='gray')
                plt.subplot(10, 7, num + 64)
                plt.axis('off')
                plt.imshow(tf.reshape(reg_zl_syn, [64, 64]), cmap='gray')
                num += 1
                if (num + 1) % 7 == 0:
                    plt.savefig(f'result/regression/Celeba_pretrain/test_{epoch}_{count}result')
                    plt.close()
                    plt.subplots(figsize=(7, 10))
                    plt.subplots_adjust(hspace=0, wspace=0)
                    num = 0
        plt.close()

        forward_psnr, forward_ssim, reg_psnr, reg_ssim = np.array(forward_psnr), np.array(forward_ssim), np.array(reg_psnr), np.array(reg_ssim)

        total_psnr = list(zip(forward_psnr, reg_psnr))
        total_ssim = list(zip(forward_ssim, reg_ssim))

        for ratio, (forward, reg) in enumerate(total_psnr):
            forward = [round(num, 1) for num in forward]
            reg = [round(num, 1) for num in reg]

            x1 = np.arange(len(forward))
            x2 = np.arange(len(reg)) - 0.2

            plt.bar(x1, forward, label='List 1', alpha=0.5)
            plt.bar(x2, reg, label='List 2', alpha=0.5)

            for i, v in enumerate(forward):
                plt.text(i, v, str(v), ha='center', va='bottom')
            for i, v in enumerate(reg):
                plt.text(i, v, str(v), ha='center', va='bottom')

            plt.xticks(x1, x1 + 1)
            # plt.legend(['forward', 'regression'], loc='upper right')
            plt.savefig(f'result/regression/Celeba_pretrain/test_{epoch}_forward_vs_reg_{2 ** (ratio + 1)}_ratio_PSNR')
            plt.close()

        for ratio, (forward, reg) in enumerate(total_ssim):
            forward = [round(num, 2) for num in forward]
            reg = [round(num, 2) for num in reg]

            x1 = np.arange(len(forward))
            x2 = np.arange(len(reg)) - 0.2

            plt.bar(x1, forward, label='List 1', alpha=0.5)
            plt.bar(x2, reg, label='List 2', alpha=0.5)

            for i, v in enumerate(forward):
                plt.text(i, v, str(v), ha='center', va='bottom')
            for i, v in enumerate(reg):
                plt.text(i, v, str(v), ha='center', va='bottom')

            plt.xticks(x1, x1 + 1)
            # plt.legend(['forward', 'regression'], loc='upper right')
            plt.savefig(f'result/regression/Celeba_pretrain/test_{epoch}_forward_vs_reg_{2 ** (ratio + 1)}_ratio_SSIM')
            plt.close()

    def main(self, pre_train=True):
        image_loss_epoch = []
        style_loss_epoch = []
        reg_loss_epoch = []
        adv_loss_epoch = []
        constrast_loss_epoch = []

        data = list(zip(self.zgHs, self.zghs, self.zgms, self.zgls, self.zghs_intepolation, self.zgms_intepolation, self.zgls_intepolation, self.zgh_neg, self.zgm_neg, self.zgl_neg, self.train_path, self.label))
        np.random.shuffle(data)
        data = list(zip(*data))
        zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zgh_neg, zgm_neg, zgl_neg, train_path, label =\
        np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), np.array(data[4]), np.array(data[5]), np.array(data[6]) , np.array(data[7]), np.array(data[8]), np.array(data[9]), np.array(data[10]), np.array(data[11])

        for epoch in range(98, self.epochs+1):
            opti = tf.keras.optimizers.Adam(1e-4)
            if epoch > 12:
                opti = tf.keras.optimizers.Adam(1e-4)

            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            reg_loss_batch = []
            adv_loss_batch = []
            constrast_loss_batch = []

            if pre_train:
                train_item = 'pre_train'
            elif epoch < 10:
                train_item = 'train_stage1'
            else:
                train_item = 'total'

            for batch in range(self.batch_num):
                batch_zgH = self.get_batch_data(zgHs, batch, self.batch_size)
                batch_zgh = self.get_batch_data(zghs, batch, self.batch_size)
                batch_zgm = self.get_batch_data(zgms, batch, self.batch_size)
                batch_zgl = self.get_batch_data(zgls, batch, self.batch_size)
                batch_label = self.get_batch_data(label, batch, self.batch_size)


                batch_zgh_intepolation = self.get_batch_data(zghs_intepolation, batch, self.batch_size)
                batch_zgm_intepolation = self.get_batch_data(zgms_intepolation, batch, self.batch_size)
                batch_zgl_intepolation = self.get_batch_data(zgls_intepolation, batch, self.batch_size)
                batch_zgh_neg = self.get_batch_data(zgh_neg, batch, self.batch_size)
                batch_zgm_neg = self.get_batch_data(zgm_neg, batch, self.batch_size)
                batch_zgl_neg = self.get_batch_data(zgl_neg, batch, self.batch_size)
                batch_high_image = self.get_batch_data(train_path, batch, self.batch_size, image=True)

                # plt.subplots(figsize=(self.batch_size, 11))
                # plt.subplots_adjust(hspace=0, wspace=0)
                # for i in range(self.batch_size):
                #     plt.subplot(11, self.batch_size, i+1)
                #     plt.axis('off')
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgH[i], [1, 200])), [64, 64]), cmap='gray')
                #
                #     plt.subplot(11, self.batch_size, i+1+1*self.batch_size)
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgh[i], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+2*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgm[i], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+3*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgl[i], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+4*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgh_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+5*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgm_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+6*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgl_intepolation[i][1], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+7*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgh_neg[i][1], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+8*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgm_neg[i][1], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+9*self.batch_size)
                #
                #     plt.imshow(tf.reshape(self.generator(tf.reshape(batch_zgl_neg[i][1], [1, 200])), [64, 64]), cmap='gray')
                #     plt.axis('off')
                #     plt.subplot(11, self.batch_size, i+1+10*self.batch_size)
                #
                #     plt.imshow(batch_high_image[i].reshape(64, 64), cmap='gray')
                #     plt.axis('off')
                # plt.show()
                # print(batch_label)

                image_loss, style_loss, reg_loss, adv_loss, constrast_loss = \
                self.train_step(batch_high_image, batch_label, batch_zgH, batch_zgh, batch_zgm, batch_zgl,
                                batch_zgh_intepolation, batch_zgm_intepolation, batch_zgl_intepolation,
                                batch_zgh_neg, batch_zgm_neg, batch_zgl_neg, opti, train=train_item)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                reg_loss_batch.append(reg_loss)
                adv_loss_batch.append(adv_loss)
                constrast_loss_batch.append(constrast_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            reg_loss_epoch.append(np.mean(reg_loss_batch))
            adv_loss_epoch.append(np.mean(adv_loss_batch))
            constrast_loss_epoch.append(np.mean(constrast_loss_batch))

            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the reg_loss is {reg_loss_epoch[-1]}')
            print(f'the adv_loss is {adv_loss_epoch[-1]}')
            print(f'the constrast_loss is {constrast_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')
            if pre_train:
                self.regression.save_weights('model_weight/pre_train_regression')
                self.plot_image_celeba(epoch)
            else:
                self.regression.save_weights('model_weight/regression_one_to_one')
                self.plot_image_AR(epoch, train=False)

            if epoch % 10 == 0:
                if pre_train:
                    filename = 'Celeba_pretrain'
                else:
                    filename = 'AR_train'

                plt.plot(image_loss_epoch)
                plt.savefig(f'result/regression/{filename}/image_loss')
                plt.close()

                plt.plot(style_loss_epoch)
                plt.savefig(f'result/regression/{filename}/style_loss')
                plt.close()

                plt.plot(reg_loss_epoch)
                plt.savefig(f'result/regression/{filename}/reg_loss')
                plt.close()

                plt.plot(adv_loss_epoch)
                plt.savefig(f'result/regression/{filename}/adv_loss')
                plt.close()

                plt.plot(constrast_loss_epoch)
                plt.savefig(f'result/regression/{filename}/constrast_loss')
                plt.close()

class test_gan_inversion():
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.regression = regression_model_with_instance()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
        # self.regression.load_weights('model_weight/pre_train_regression')
        self.generator.load_weights('model_weight/zd_zg_distillation_generator')
        self.discriminator.load_weights('model_weight/patch_d')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

    def prepare_all_data(self, multiple=True):
        high_images, test_path = [], []
        if multiple:
            path_AR_syn = '/home/bosen/PycharmProjects/Datasets/AR_test/'
            ID = [f'ID0{i}' if i < 10 else f'ID{i}' for i in range(1, 22)]

            for id_th, id in enumerate(ID):
                for num, filename in enumerate(os.listdir(path_AR_syn + id)):
                    if 50 > num > 20:
                        image = cv2.imread(path_AR_syn + id + '/' + filename, 0) / 255
                        high_images.append(cv2.resize(image, (64, 64), cv2.INTER_CUBIC))
                        test_path.append(path_AR_syn + id + '/' + filename)
            high_images, test_path = np.array(high_images), np.array(test_path)
            return high_images, test_path
        else:
            path_AR_syn = "/disk2/DCGAN_yu/CK1/ARtest/"
            for num, filename in enumerate(os.listdir(path_AR_syn)):
                image = cv2.imread(path_AR_syn + '/' + filename, 0) / 255
                high_images.append(cv2.resize(image, (64, 64), cv2.INTER_CUBIC))
                test_path.append(path_AR_syn + '/' + filename)
            high_images, test_path = np.array(high_images), np.array(test_path)
            return high_images, test_path

    def get_batch_data(self, data_path, reso):
        low_images, corresponding_code = [], []
        for path in data_path:
            image = cv2.imread(path, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            low_image = cv2.resize(cv2.resize(image, (reso, reso), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z = self.encoder(low_image.reshape(1, 64, 64, 1))
            zg, _, _ = self.ztozg(z)
            low_images.append(low_image)
            corresponding_code.append(tf.reshape(zg, [200]))
        low_images, corresponding_code = np.array(low_images), np.array(corresponding_code)
        return low_images, corresponding_code

    def perceptual_loss(self, real_low, fake_high, reso):
        synthesis_image = []
        for i in range(fake_high.shape[0]):
            fake = tf.image.resize(fake_high[i], [reso, reso], method='bicubic')
            synthesis_image.append(tf.image.resize(fake, [64, 64], method='bicubic'))
        synthesis_image = tf.cast(synthesis_image, dtype=tf.float32)

        real_low, fake_low = tf.cast(real_low, dtype="float32"), tf.cast(synthesis_image, dtype="float32")
        real_low = tf.image.grayscale_to_rgb(real_low)
        fake_low = tf.image.grayscale_to_rgb(fake_low)
        real_feature = self.feature_extraction(real_low)
        fake_feature = self.feature_extraction(fake_low)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def image_loss(self, real_low, fake_high, reso):
        synthesis_image = []
        for i in range(fake_high.shape[0]):
            fake = tf.image.resize(fake_high[i], [reso, reso], method='bicubic')
            synthesis_image.append(tf.image.resize(fake, [64, 64], method='bicubic'))
        synthesis_image = tf.cast(synthesis_image, dtype=tf.float32)
        image_loss = tf.cast(tf.reduce_mean(tf.square(real_low - synthesis_image)), dtype=tf.float32)
        return image_loss

    def adv_loss(self, fake_high):
        synthesis_score = self.discriminator(fake_high)
        adv_loss = tf.reduce_mean(tf.square(synthesis_score - 1))
        return adv_loss

    def zg_inversion_step(self, real_low, code, reso):
        real_low = real_low.reshape(-1, 64, 64, 1)
        code = tf.cast(code, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(code)
            fake_high = tf.reshape(self.generator(code), [-1, 64, 64, 1])

            rec_loss = 10 * self.image_loss(real_low, fake_high, reso)
            style_loss = self.perceptual_loss(real_low, fake_high, reso)
            adv_loss = 0.1 * self.adv_loss(fake_high)

            total_loss = rec_loss + style_loss + adv_loss
        gradient_code = code_tape.gradient(total_loss, code)
        code = code - self.learning_rate * gradient_code
        return rec_loss, style_loss, adv_loss, code

    def multiple_inversion(self):
        # cls_test_inversion_data_path = 'cls_data/test_data_inversion/'
        # for id in os.listdir(cls_test_inversion_data_path):
        #     for filename in os.listdir(cls_test_inversion_data_path + id ):
        #         os.remove(cls_test_inversion_data_path + id + '/' + filename)

        rec_loss_epoch = []
        style_loss_epoch = []
        adv_loss_epoch = []
        high_images, test_path = self.prepare_all_data()
        low1_images, low1_code = self.get_batch_data(test_path, 32)
        low2_images, low2_code = self.get_batch_data(test_path, 16)
        low3_images, low3_code = self.get_batch_data(test_path, 8)

        low_image = [low1_images, low2_images, low3_images]
        forward_code = [low1_code, low2_code, low3_code]
        data = list(zip(low_image, forward_code))

        for num, (test_low, test_code) in enumerate(data):

            if num == 0: reso = 32
            elif num == 1: reso = 16
            else: reso = 8

            for i in range(3):
                print(int(i*test_low.shape[0]/3), int((i*test_low.shape[0]/3) + test_low.shape[0]/3))
                high = high_images[int(i*test_low.shape[0]/3): int((i*test_low.shape[0]/3) + test_low.shape[0]/3)]
                path = test_path[int(i*test_low.shape[0]/3): int((i*test_low.shape[0]/3) + test_low.shape[0]/3)]
                low = test_low[int(i*test_low.shape[0]/3): int((i*test_low.shape[0]/3) + test_low.shape[0]/3)]
                forward_code = test_code[int(i*test_low.shape[0]/3): int((i*test_low.shape[0]/3) + test_low.shape[0]/3)]
                code = forward_code
                print(high.shape, path.shape, low.shape, forward_code.shape)

                res_rec_loss = 10
                for epoch in range(1, self.epochs+1):
                    start = time.time()
                    rec_loss, style_loss, adv_loss, code = self.zg_inversion_step(low, code, reso=reso)
                    rec_loss_epoch.append(rec_loss)
                    style_loss_epoch.append(style_loss)
                    adv_loss_epoch.append(adv_loss)
                    print("______________________________________")
                    print(f"the epoch is {epoch}")
                    print(f"the mse_loss is {rec_loss}")
                    print(f"the perceptual_loss is {style_loss}")
                    print(f'the adv loss is {adv_loss}')
                    print(f"the new_code is {code[0][0:10]}")
                    print("the spend time is %s second" % (time.time() - start))

                    #draw sample
                    if rec_loss < res_rec_loss and epoch > 400:
                        print(path[num], path[num][47: 49], path[num][50:])
                        for num, zg in enumerate(code):
                            syn_image = tf.reshape(self.generator(tf.reshape(zg, [1, 200])), [64, 64])
                            cv2.imwrite(f'cls_data/test_data_inversion/ID{int(path[num][47: 49]) + 90}/{path[num][50:]}_{reso}_resolution.jpg',
                                        np.array(syn_image).reshape(64, 64) * 255)


                        res_rec_loss = rec_loss
                        # np.savetxt(f"result/inversion_code/cls_test_data/final_test_zg_{self.type}_inv_result.csv", code, delimiter=",")
                        res = 0
                        plt.subplots(figsize=(7, 4))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        for count in [0, 30, 60, 90, 120, 150, 180]:
                            plt.subplot(4, 7, res + 1)
                            plt.axis('off')
                            plt.imshow(high[count], cmap='gray')

                            plt.subplot(4, 7, res + 8)
                            plt.axis('off')
                            plt.imshow(tf.reshape(low[count], [64, 64]), cmap='gray')

                            plt.subplot(4, 7, res + 15)
                            plt.axis('off')
                            plt.imshow(tf.reshape(self.generator(tf.reshape(forward_code[count], [1, 200])), [64, 64]), cmap='gray')

                            plt.subplot(4, 7, res + 22)
                            plt.axis('off')
                            plt.imshow(tf.reshape(self.generator(tf.reshape(code[count], [1, 200])), [64, 64]), cmap='gray')
                            res += 1
                            if (res) % 7 == 0:
                                plt.savefig(f'result/inversion/{reso}_resolution_{i}_th.jpg')
                                plt.close()
                                print('update the latent code')
                                plt.subplots(figsize=(7, 4))
                                plt.subplots_adjust(wspace=0, hspace=0)
                                res = 0
                        plt.close()

    def single_inversion(self):
        rec_loss_epoch = []
        style_loss_epoch = []
        adv_loss_epoch = []
        high_images, test_path = self.prepare_all_data(multiple=False)
        low1_images, low1_code = self.get_batch_data(test_path, 32)
        low2_images, low2_code = self.get_batch_data(test_path, 16)
        low3_images, low3_code = self.get_batch_data(test_path, 8)

        low_image = [low1_images, low2_images, low3_images]
        forward_code = [low1_code, low2_code, low3_code]
        data = list(zip(low_image, forward_code))

        for num, (test_low, test_code) in enumerate(data):
            if num == 0:
                reso = 32
            elif num == 1:
                reso = 16
            else:
                reso = 8

            print(high_images.shape, test_path.shape, test_low.shape, test_code.shape)
            code = test_code
            res_rec_loss = 10
            for epoch in range(1, self.epochs + 1):
                start = time.time()
                rec_loss, style_loss, adv_loss, code = self.zg_inversion_step(test_low, code, reso=reso)
                rec_loss_epoch.append(rec_loss)
                style_loss_epoch.append(style_loss)
                adv_loss_epoch.append(adv_loss)
                print("______________________________________")
                print(f"the epoch is {epoch}")
                print(f"the mse_loss is {rec_loss}")
                print(f"the perceptual_loss is {style_loss}")
                print(f'the adv loss is {adv_loss}')
                print(f"the new_code is {code[0][0:10]}")
                print("the spend time is %s second" % (time.time() - start))

                # draw sample
                if (rec_loss < res_rec_loss and epoch > 780) or epoch == 1 or epoch == 100 or epoch == 500:
                    res_rec_loss = rec_loss
                    plt.subplots(figsize=(21, 4))
                    plt.subplots_adjust(wspace=0, hspace=0)
                    psnr_inte, psnr_for, psnr_inv = [], [], []
                    ssim_inte, ssim_for, ssim_inv = [], [], []

                    for i in range(21):
                        plt.subplot(4, 21, i + 1)
                        plt.axis('off')
                        plt.imshow(high_images[i], cmap='gray')

                        plt.subplot(4, 21, i + 22)
                        plt.axis('off')
                        plt.imshow(tf.reshape(test_low[i], [64, 64]), cmap='gray')

                        plt.subplot(4, 21, i + 43)
                        plt.axis('off')
                        plt.imshow(tf.reshape(self.generator(tf.reshape(forward_code[num][i], [1, 200])), [64, 64]), cmap='gray')

                        plt.subplot(4, 21, i + 64)
                        plt.axis('off')
                        plt.imshow(tf.reshape(self.generator(tf.reshape(code[i], [1, 200])), [64, 64]), cmap='gray')

                        psnr_inte.append(tf.image.psnr(tf.cast(high_images[i].reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(test_low[i].reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0])
                        psnr_for.append(tf.image.psnr(tf.cast(high_images[i].reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(self.generator(tf.reshape(forward_code[num][i], [1, 200])), dtype=tf.float32), max_val=1)[0])
                        psnr_inv.append(tf.image.psnr(tf.cast(high_images[i].reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(self.generator(tf.reshape(code[i], [1, 200])), dtype=tf.float32), max_val=1)[0])

                        ssim_inte.append(tf.image.ssim(tf.cast(high_images[i].reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(test_low[i].reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0])
                        ssim_for.append(tf.image.ssim(tf.cast(high_images[i].reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(self.generator(tf.reshape(forward_code[num][i], [1, 200])),dtype=tf.float32), max_val=1)[0])
                        ssim_inv.append(tf.image.ssim(tf.cast(high_images[i].reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(self.generator(tf.reshape(code[i], [1, 200])), dtype=tf.float32), max_val=1)[0])
                    plt.savefig(f'result/inversion/single_inversion/test_{reso}_resolution_{epoch}.jpg')
                    print(reso)
                    plt.close()
                    print(psnr_inte, ssim_inte)
                    print(psnr_for, ssim_for)
                    print(psnr_inv, ssim_inv)


class reid_image_classifier():
    def __init__(self, epochs, batch_num, batch_size):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.opti = tf.keras.optimizers.Adam(5e-5)
        self.cls = classifier()
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.regression = regression_model()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.cls.load_weights('model_weight/reid_cls')
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
        self.regression.load_weights('model_weight/regression2')
        self.generator.load_weights('model_weight/zd_zg_distillation_generator')

        # for layer in self.cls.layers:
        #     print(layer.name)

        self.last_conv_layer_name = 'conv2d'
        self.network_layer_name = ['max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'flatten',
                                   'dense', 'dropout', 'dense_1']

        # self.last_conv_layer_name = 'conv2d_2'
        # self.network_layer_name = ['max_pooling2d_2', 'flatten', 'dense_13', 'dropout', 'flatten', 'dense_14']

    def prepare_train_data(self):
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

        train_data, test_data, train_label, test_label = [], [], [], []

        for id in os.listdir(path_AR_syn_train):
            for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_train + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    # low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    # low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    # low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    #
                    # z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)
                    # zg1, _, _ = self.ztozg(z1)
                    # zg2, _, _ = self.ztozg(z2)
                    # zg3, _, _ = self.ztozg(z3)
                    # syn1, syn2, syn3 = self.generator(zg1), self.generator(zg2), self.generator(zg3)
                    #
                    train_data.append(image)
                    # train_data.append(tf.reshape(syn1, [64, 64]))
                    # train_data.append(tf.reshape(syn2, [64, 64]))
                    # train_data.append(tf.reshape(syn3, [64, 64]))

                    # for i in range(4):
                    #     train_label.append(tf.one_hot(int(id[2:]) - 1, 111))
                    train_label.append(tf.one_hot(int(id[2:]) - 1, 111))
                else:
                    break

        for id in os.listdir(path_AR_syn_test):
            for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    # low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    # low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    # low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    #
                    # z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)
                    # zg1, _, _ = self.ztozg(z1)
                    # zg2, _, _ = self.ztozg(z2)
                    # zg3, _, _ = self.ztozg(z3)
                    # syn1, syn2, syn3 = self.generator(zg1), self.generator(zg2), self.generator(zg3)
                    train_data.append(image)
                    # train_data.append(tf.reshape(syn1, [64, 64]))
                    # train_data.append(tf.reshape(syn2, [64, 64]))
                    # train_data.append(tf.reshape(syn3, [64, 64]))

                    # for i in range(4):
                    #     train_label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
                    train_label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
                else:
                    break
                        
        training_data = list(zip(train_data, train_label))
        np.random.shuffle(training_data)
        training_data = list(zip(*training_data))
        return np.array(training_data[0]), np.array(training_data[1])

    def get_batch_data(self, data, batch_idx, batch_size, image=False):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        if image:
            return np.array(train_data).reshape(-1, 64, 64, 1)
        else:
            return np.array(train_data)

    def gradcam_heatmap_mutiple(self, img_array, model, last_conv_layer_name, network_layer_name, label, corresponding_label=True):
        label = np.argmax(label, axis=-1)
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

        network_input = Input(shape=last_conv_layer.output.shape[1:])
        x = network_input
        for layer_name in network_layer_name:
            x = model.get_layer(layer_name)(x)
        network_model = Model(network_input, x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = network_model(last_conv_layer_output)
            if corresponding_label:
                pred_index = tf.constant(label, dtype=tf.int64)
            else:
                pred_index = np.argmax(preds, axis=-1)
            class_channel = tf.gather(preds, pred_index, axis=-1, batch_dims=1)
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        pooled_gradsa = tf.tile(tf.reshape(pooled_grads, [pooled_grads.shape[0], 1, 1, pooled_grads.shape[1]]), [1, last_conv_layer_output.shape[1], last_conv_layer_output.shape[2], 1])
        heatmap = last_conv_layer_output * pooled_gradsa
        heatmap = tf.reduce_sum(heatmap, axis=-1)
        heatmap_min, heatmap_max = [], []
        for num in range(heatmap.shape[0]):
            heatmap_min.append(np.min(heatmap[num]))
            heatmap_max.append(np.max(heatmap[num]))
        heatmap_min, heatmap_max = tf.cast(heatmap_min, dtype=tf.float32), tf.cast(heatmap_max, dtype=tf.float32)
        heatmap_min = tf.tile(tf.reshape(heatmap_min, [heatmap_min.shape[0], 1, 1]), [1, heatmap.shape[1], heatmap.shape[2]])
        heatmap_max = tf.tile(tf.reshape(heatmap_max, [heatmap_max.shape[0], 1, 1]), [1, heatmap.shape[1], heatmap.shape[2]])
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-20)

        heatmap_gray = tf.cast(heatmap, dtype=tf.float32)
        heatmap = np.uint8(255 * heatmap)

        cmap = cm.get_cmap("jet")
        cmap_colors = cmap(np.arange(256))[:, :3]
        cmap_heatmap = cmap_colors[heatmap]
        cmap_heatmap = tf.image.resize(cmap_heatmap, [64, 64], method='bicubic')
        heatmap_gray = tf.image.resize(tf.reshape(heatmap_gray, [-1, 64, 64, 1]), [64, 64], method='bicubic')
        return cmap_heatmap, heatmap_gray

    def cross_entropy_loss(self, label, pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(label, pred)

    def inverse_cross_entropy_loss(self, label, pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        gt = tf.constant([[1, 0]], dtype=tf.float32)
        gt = tf.tile(gt, [pred.shape[0], 1])
        label_index = np.argmax(label, axis=-1)
        pred_value = tf.gather(pred, label_index, axis=-1, batch_dims=1)
        pred_value = tf.expand_dims(pred_value, axis=-1)
        exp_tensor = 1 - pred_value
        pred_value = tf.concat([exp_tensor, pred_value], axis=-1)
        return cce(gt, pred_value)

    def train_step(self, image, label, reid=False):
        with tf.GradientTape() as tape:
            att, attention_map = self.gradcam_heatmap_mutiple(image, self.cls, self.last_conv_layer_name, self.network_layer_name, label, corresponding_label=True)
            inverse_image = image * (1 - attention_map)

            pred = self.cls(image)
            pred_inv = self.cls(inverse_image)
            ce_loss = self.cross_entropy_loss(label, pred)
            inv_ce_loss = self.inverse_cross_entropy_loss(label, pred_inv)

            acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred, axis=-1))

            total_loss = ce_loss
            reid_loss = inv_ce_loss + ce_loss

        if reid:
            grads = tape.gradient(reid_loss, self.cls.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.cls.trainable_variables))
        else:
            grads = tape.gradient(total_loss, self.cls.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.cls.trainable_variables))
        return ce_loss, inv_ce_loss, acc

    def main(self, reid=False):
        ce_epoch = []
        inv_ce_epoch = []
        acc_epoch = []
        train_path, train_label = self.prepare_train_data()
        print(train_path.shape, train_label.shape)

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            ce_loss_batch = []
            inv_ce_loss_batch = []
            acc_batch = []

            for batch in range(self.batch_num):
                batch_train_image = self.get_batch_data(train_path, batch, self.batch_size, image=True)
                batch_train_label = self.get_batch_data(train_label, batch, self.batch_size, image=False)

                ce_loss, inv_ce_loss, acc = self.train_step(batch_train_image, batch_train_label, reid=reid)
                ce_loss_batch.append(ce_loss)
                acc_batch.append(acc)
                inv_ce_loss_batch.append(inv_ce_loss)

            ce_epoch.append(np.mean(ce_loss_batch))
            acc_epoch.append(np.mean(acc_batch))
            inv_ce_epoch.append(np.mean(inv_ce_loss_batch))

            print(f'the epoch is {epoch}')
            print(f'the ce_loss is {ce_epoch[-1]}')
            print(f'the accuracy is {acc_epoch[-1]}')
            print(f'the inv_ce_loss is {inv_ce_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')
            if reid:
                self.cls.save_weights('model_weight/reid_cls')
            else:
                self.cls.save_weights('model_weight/cls')

        if reid:
            filename = 'reid_cls'
        else:
            filename = 'cls'

        plt.plot(ce_epoch)
        plt.savefig(f'result/cls/{filename}/ce_loss')
        plt.close()

        plt.plot(acc_epoch)
        plt.savefig(f'result/cls/{filename}/acc')
        plt.close()

        plt.plot(inv_ce_epoch)
        plt.savefig(f'result/cls/{filename}/inv_ce_loss')
        plt.close()
        # self.forward_confusion_matrix(train=True, reid=reid)
        self.forward_confusion_matrix(train=False, reid=reid)
        self.inversion_confusion_matrix(reid=reid)

    def forward_confusion_matrix(self, train=True, reid=False):
        if train:
            path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_train/'

        else:
            path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

        label = []
        pred_low1, pred_low2, pred_low3 = [], [], []
        pred_forward_low1, pred_forward_low2, pred_forward_low3 = [], [], []

        for id in os.listdir(path_AR_syn_test):
            for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
                if 50 > count > 20:
                    label.append(int(id[2:])-1+90)
                    image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    low1_test = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_test = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_test = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                    z1, z2, z3 = self.encoder(low1_test), self.encoder(low2_test), self.encoder(low3_test)
                    zg1, _, _ = self.ztozg(z1)
                    zg2, _, _ = self.ztozg(z2)
                    zg3, _, _ = self.ztozg(z3)

                    # zg1 = self.regression(zg1)
                    # zg2 = self.regression(zg2)
                    # zg3 = self.regression(zg3)

                    forward_low1_test = self.generator(zg1)
                    forward_low2_test = self.generator(zg2)
                    forward_low3_test = self.generator(zg3)

                    pred_low1.append(tf.reshape(self.cls(low1_test), [111]))
                    pred_low2.append(tf.reshape(self.cls(low2_test), [111]))
                    pred_low3.append(tf.reshape(self.cls(low3_test), [111]))
                    pred_forward_low1.append(tf.reshape(self.cls(forward_low1_test), [111]))
                    pred_forward_low2.append(tf.reshape(self.cls(forward_low2_test), [111]))
                    pred_forward_low3.append(tf.reshape(self.cls(forward_low3_test), [111]))

        confusion_matrix_low1 = metrics.confusion_matrix(label, np.argmax(pred_low1, axis=-1))
        accuracy_low1 = accuracy_score(label, np.argmax(pred_low1, axis=-1))

        confusion_matrix_low2 = metrics.confusion_matrix(label, np.argmax(pred_low2, axis=-1))
        accuracy_low2 = accuracy_score(label, np.argmax(pred_low2, axis=-1))

        confusion_matrix_low3 = metrics.confusion_matrix(label, np.argmax(pred_low3, axis=-1))
        accuracy_low3 = accuracy_score(label, np.argmax(pred_low3, axis=-1))

        confusion_matrix_forward_low1 = metrics.confusion_matrix(label, np.argmax(pred_forward_low1, axis=-1))
        accuracy_forward_low1 = accuracy_score(label, np.argmax(pred_forward_low1, axis=-1))

        confusion_matrix_forward_low2 = metrics.confusion_matrix(label, np.argmax(pred_forward_low2, axis=-1))
        accuracy_forward_low2 = accuracy_score(label, np.argmax(pred_forward_low2, axis=-1))

        confusion_matrix_forward_low3 = metrics.confusion_matrix(label, np.argmax(pred_forward_low3, axis=-1))
        accuracy_forward_low3 = accuracy_score(label, np.argmax(pred_forward_low3, axis=-1))

        confusion_matrix = [confusion_matrix_low1, confusion_matrix_low2, confusion_matrix_low3, confusion_matrix_forward_low1, confusion_matrix_forward_low2, confusion_matrix_forward_low3]
        accuracy = [accuracy_low1, accuracy_low2, accuracy_low3, accuracy_forward_low1, accuracy_forward_low2, accuracy_forward_low3]

        for num, name in enumerate(['2-ratio', '4-ratio', '8-ratio', '2-ratio-syn', '4-ratio-syn', '8-ratio-syn']):
            plt.title(f'Acc = {str(accuracy[num])[0:5]}')
            plt.axis('off')
            sns.heatmap(confusion_matrix[num], vmin=0, vmax=30)
            if reid:
                if train:
                    plt.savefig(f'result/cls/reid_cls/train_confusion_matrix_{name}')
                else:
                    plt.savefig(f'result/cls/reid_cls/test_confusion_matrix_{name}')
            else:
                if train:
                    plt.savefig(f'result/cls/cls/train_confusion_matrix_{name}')
                else:
                    plt.savefig(f'result/cls/cls/tes_confusionmatrix_{name}')
            plt.close()

    def inversion_confusion_matrix(self, reid=False):
        path = 'cls_data/test_data_inversion/'

        label1, label2, label3 = [], [], []
        pred1, pred2, pred3 = [], [], []

        for id in os.listdir(path):
            for count, filename in enumerate(os.listdir(path + id)):
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                if '32_resolution' in filename:
                    label1.append(int(id[2:]) - 1)
                    pred1.append(tf.reshape(self.cls(image.reshape(1, 64, 64, 1)), [111]))

                if '16_resolution' in filename:
                    label2.append(int(id[2:]) - 1)
                    pred2.append(tf.reshape(self.cls(image.reshape(1, 64, 64, 1)), [111]))

                if '8_resolution' in filename:
                    label3.append(int(id[2:]) - 1)
                    pred3.append(tf.reshape(self.cls(image.reshape(1, 64, 64, 1)), [111]))

        confusion_matrix1 = metrics.confusion_matrix(label1, np.argmax(pred1, axis=-1))
        accuracy1 = accuracy_score(label1, np.argmax(pred1, axis=-1))

        confusion_matrix2 = metrics.confusion_matrix(label2, np.argmax(pred2, axis=-1))
        accuracy2 = accuracy_score(label2, np.argmax(pred2, axis=-1))

        confusion_matrix3 = metrics.confusion_matrix(label3, np.argmax(pred3, axis=-1))
        accuracy3 = accuracy_score(label3, np.argmax(pred3, axis=-1))



        confusion_matrix = [confusion_matrix1, confusion_matrix2, confusion_matrix3]
        accuracy = [accuracy1, accuracy2, accuracy3]

        for num, name in enumerate(['2-ratio', '4-ratio', '8-ratio']):
            plt.title(f'Acc = {str(accuracy[num])[0:5]}')
            plt.axis('off')
            sns.heatmap(confusion_matrix[num], vmin=0, vmax=30)
            if reid:
                plt.savefig(f'result/cls/reid_cls/test_confusion_matrix_{name}_inv')
                plt.close()
            else:
                plt.savefig(f'result/cls/cls/test_confusion_matrix_{name}_inv')
                plt.close()


if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    #ae item.
    # ae = AE(epochs=350, batch_size=30, batch_num=138)
    # ae.training()
    # for i in [1, 2, 3]:
    #     ae.validate_AE_trained_well(i, train=True)
    #     ae.validate_AE_trained_well(i, train=False)


    #patchgan item.
    # patch_gan = PatchGAN(epochs=150, batch_size=30, batch_num=138)
    # # patch_gan.training()
    # for i in [1, 2, 3]:
    #     patch_gan.validate_G_trained_well(i, train=True)
    #     patch_gan.validate_G_trained_well(i, train=False)
    # patch_gan.validate_D_trained_well()


    #distillation item.
    # zd_zg_distillation = zd_zg_distillation(epochs=70, batch_size=30, batch_num=138)
    # zd_zg_distillation.training()
    # zd_zg_distillation.different_resolution_synthesis_result(train=True)
    # zd_zg_distillation.different_resolution_synthesis_result(train=False)
    # for i in [1, 2, 3]:
    #     zd_zg_distillation.validate_G_trained_well(i, train=True)
    #     zd_zg_distillation.validate_G_trained_well(i, train=False)



    # #test contrast loss min&max situation.
    # reg = Regression(epochs=100, batch_size=15, batch_num=138)
    # zregHs, zreghs, zregms, zregls, _, _, _, zregh_neg, zregm_neg, zregl_neg, _ = reg.AR_regression_training_data(contrast=True)
    # print(zregHs.shape, zreghs.shape, zregms.shape, zregls.shape, zregh_neg.shape, zregm_neg.shape, zregl_neg.shape)
    # reg.constrast_loss(zregHs,  zreghs, zregms, zregls, zregh_neg, zregm_neg, zregl_neg)
    # zregh_neg = -tf.tile(tf.reshape(zregHs, [-1, 1, 200]), [1, 3, 1])
    # zregm_neg = -tf.tile(tf.reshape(zregHs, [-1, 1, 200]), [1, 3, 1])
    # zregl_neg = -tf.tile(tf.reshape(zregHs, [-1, 1, 200]), [1, 3, 1])
    # #min situation.
    # reg.constrast_loss(zregHs,  zregHs, zregHs, zregHs, zregh_neg, zregm_neg, zregl_neg)
    # #max situation.
    # reg.constrast_loss(zregHs, -zregHs, -zregHs, -zregHs, -zregh_neg, -zregm_neg, -zregl_neg)

    # reg = Regression(epochs=150, batch_size=15, batch_num=138)
    # reg.compare_multiple_method()

    # cls = reid_image_classifier(12, batch_size=20, batch_num=111)
    # cls.forward_confusion_matrix(train=False, reid=True)

    # zg_inversion = test_gan_inversion(epochs=500, learning_rate=20000)
    # zg_inversion.multiple_inversion()


    # zg_inversion = test_gan_inversion(epochs=800, learning_rate=5000)
    # zg_inversion.single_inversion()

    path_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
    path_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
    path_celeba = "/home/bosen/PycharmProjects/SRGAN_learning_based_inversion/celeba_train/"

    path = "/disk2/bosen/end_to_end_training/AR_train_data/"
    for id in os.listdir(path_train):
        for count, filename in enumerate(os.listdir(path_train + id)):
            if count < 20:
                image = cv2.imread(path_train + id + '/' + filename, 0)
                cv2.imwrite(f"/disk2/bosen/end_to_end_training/AR_train_data/ID{int(id[2:])}/{filename}.jpg", image)





























