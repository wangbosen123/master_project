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
        self.encoder = encoder()
        self.decoder = decoder()
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
        self.encoder.load_weights('weights/encoder')
        self.decoder.load_weights('weights/decoder')

        #set the data path
        self.train_path, self.train_label, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.train_label.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        path_AR_syn_train = '/disk2/bosen/Datasets/AR_train/'
        path_AR_syn_test = '/disk2/bosen/Datasets/AR_test/'
        path_AR_real_train = "/disk2/bosen/Datasets/AR_original_alignment_train90/"
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


        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if '11_test' in filename:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)

        train_path, train_label, test_path1, test_path2, test_path3 = np.array(train_path), np.array(train_label), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        train_data = list(zip(train_path, train_label))
        np.random.shuffle(train_data)
        train_data = list(zip(*train_data))
        return np.array(train_data[0]), np.array(train_data[1]), test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size):
        train_image = []
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
            train_image.append(image), train_image.append(low1_image), train_image.append(low2_image), train_image.append(low3_image)

        train_image = np.array(train_image).reshape(-1, 64, 64, 1)
        return train_image

    def style_loss(self, real, fake):
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = self.feature_extraction(real)
        fake_feature = self.feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def train_step(self, low_images, opti, train=True):
        with tf.GradientTape() as tape:

            z = self.encoder(low_images)
            gen_images = self.decoder(z)
            image_loss = 10 * tf.reduce_mean(tf.square(low_images - gen_images))
            style_loss = 10 * self.style_loss(low_images, gen_images)
            total_loss = image_loss + style_loss

        if train:
            grads = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
            opti.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))
        return image_loss, style_loss

    def training(self):
        image_loss_epoch = []
        style_loss_epoch = []
        opti = tf.keras.optimizers.Adam(1e-5)

        for epoch in range(137, self.epochs+1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []

            if epoch > 300:
                opti = tf.keras.optimizers.Adam(1e-5)

            for batch in range(self.batch_num):
                train_image_batch = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                image_loss, style_loss = self.train_step(train_image_batch, opti=opti, train=True)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.encoder.save_weights('weights/encoder')
            self.decoder.save_weights('weights/decoder')
            self.plot_image(epoch, self.test_path1, data_name='test_21ID')

            if epoch == self.epochs:
                self.plot_image(epoch, self.test_path2, data_name='train_90ID')
                self.plot_image(epoch, self.test_path3, data_name='train_celeba')

            plt.plot(image_loss_epoch)
            plt.savefig('result/AE/image_loss')
            plt.close()

            plt.plot(style_loss_epoch)
            plt.savefig('result/AE/style_loss')
            plt.close()

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

            gen_image = self.decoder(z)
            gen1_image = self.decoder(z1)
            gen2_image = self.decoder(z2)
            gen3_image = self.decoder(z3)

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
                plt.subplots(figsize=(7, 8))
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
        syn_image = self.decoder(z)
        plt.subplot(7, 4, 9)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        z = self.encoder(nn_face_low1[1].reshape(1, 64, 64, 1))
        syn_image = self.decoder(z)
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
        syn_image = self.decoder(z)
        plt.subplot(7, 4, 17)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        plt.subplot(7, 4, 18)
        plt.axis('off')
        plt.imshow(np.zeros((64, 64)), cmap='gray')
        z = self.encoder(nn_face_low2[1].reshape(1, 64, 64, 1))
        syn_image = self.decoder(z)
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
        syn_image = self.decoder(z)
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
        syn_image = self.decoder(z)
        plt.subplot(7, 4, 28)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_image, [64, 64]), cmap='gray')
        if train:
            plt.savefig(f'result/AE/train_id{id_index}_similarity_syn')
            plt.close()
        else:
            plt.savefig(f'result/AE/test_id{id_index}_similarity_syn')
            plt.close()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    AE = AE(epochs=300, batch_num=69, batch_size=60)
    AE.training()
