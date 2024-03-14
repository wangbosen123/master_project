from GAN import *
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import dlib
import re


def repeat_overall_structure():
    def forward(image, times):
        for i in range(times):
            z = encoder(image)
            _, _, zreg = reg(z)
            image = generator(zreg)
        return image

    global encoder
    global generator
    global reg

    encoder = encoder()
    generator = generator()
    reg = regression()

    encoder.load_weights('weights/encoder')
    generator.load_weights('weights/generator2')
    reg.load_weights('weights/reg')

    path_AR_syn_test = '/disk2/bosen/Datasets/AR_test/'
    path = []
    for id in os.listdir(path_AR_syn_test):
        for filename in os.listdir(path_AR_syn_test + id):
            if '11_test' in filename:
                path.append(path_AR_syn_test + id + '/' + filename)
    path = np.array(path)

    plt.subplots(figsize=(7, 16))
    plt.subplots_adjust(hspace=0, wspace=0)
    count = 0
    for num, filename in enumerate(path):
        image = cv2.imread(filename, 0) / 255
        image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        syn_high_1, syn_low1_1, syn_low2_1, syn_low3_1 = forward(image.reshape(1, 64, 64, 1), 1), forward(low1_image.reshape(1, 64, 64, 1), 1), forward(low2_image.reshape(1 ,64, 64, 1), 1), forward(low3_image.reshape(1, 64, 64, 1), 1)
        syn_high_2, syn_low1_2, syn_low2_2, syn_low3_2 = forward(image.reshape(1, 64, 64, 1), 2), forward(low1_image.reshape(1, 64, 64, 1), 2), forward(low2_image.reshape(1 ,64, 64, 1), 2), forward(low3_image.reshape(1, 64, 64, 1), 2)
        syn_high_3, syn_low1_3, syn_low2_3, syn_low3_3 = forward(image.reshape(1, 64, 64, 1), 3), forward(low1_image.reshape(1, 64, 64, 1), 3), forward(low2_image.reshape(1 ,64, 64, 1), 3), forward(low3_image.reshape(1, 64, 64, 1), 3)


        plt.subplot(16, 7, count + 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')

        plt.subplot(16, 7, count + 8)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_high_1, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 15)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_high_2, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 22)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_high_3, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 29)
        plt.axis('off')
        plt.imshow(low1_image, cmap='gray')

        plt.subplot(16, 7, count + 36)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low1_1, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 43)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low1_2, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 50)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low1_3, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 57)
        plt.axis('off')
        plt.imshow(low2_image, cmap='gray')

        plt.subplot(16, 7, count + 64)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low2_1, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 71)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low2_2, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 78)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low2_3, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 85)
        plt.axis('off')
        plt.imshow(low3_image, cmap='gray')

        plt.subplot(16, 7, count + 92)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low3_1, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 99)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low3_2, [64, 64]), cmap='gray')

        plt.subplot(16, 7, count + 106)
        plt.axis('off')
        plt.imshow(tf.reshape(syn_low3_3, [64, 64]), cmap='gray')


        count += 1

        if (num + 1) % 7 == 0:
            if reg:
                # plt.savefig(f'result/GAN/after_reg_{data_name}_{epoch}_{num + 1}image')
                plt.show()
            else:
                # plt.savefig(f'result/GAN/before_reg_{data_name}_{epoch}_{num + 1}image')
                plt.show()

            plt.close()
            plt.subplots(figsize=(7, 16))
            plt.subplots_adjust(hspace=0, wspace=0)
            count = 0

# def record_min_max_repeat_reg(train=True):
#     global encoder
#     global reg
#     global generator
#     global discriminator
#
#     encoder = encoder()
#     reg = regression()
#     generator = generator()
#     discriminator = discriminator()
#
#     encoder.load_weights('weights/encoder')
#     reg.load_weights('weights/reg')
#     generator.load_weights('weights/generator2')
#     discriminator.load_weights('weights/discriminator2')
#
#     def down_image(image, ratio):
#         if ratio == 1:
#             return tf.cast(image, dtype=tf.float32)
#         if ratio == 2:
#             down_syn = tf.image.resize(image, [32, 32], method='bicubic')
#             down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
#             return down_syn
#         elif ratio == 4:
#             down_syn = tf.image.resize(image, [16, 16], method='bicubic')
#             down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
#             return down_syn
#         elif ratio == 8:
#             down_syn = tf.image.resize(image, [8, 8], method='bicubic')
#             down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
#             return down_syn
#
#     def style_loss(real, fake):
#         feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#         real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
#         real = tf.image.grayscale_to_rgb(real)
#         fake = tf.image.grayscale_to_rgb(fake)
#
#         real_feature = feature_extraction(real)
#         fake_feature = feature_extraction(fake)
#         distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
#         return distance
#
#
#     if train: path = '/disk2/bosen/Datasets/AR_train/'
#     else: path = '/disk2/bosen/Datasets/AR_test/'
#
#     def search(ratio, loss_type):
#         database_latent, database_gt, database_image = [], [], []
#         for id_num, id in enumerate(os.listdir(path)):
#             for file_num, filename in enumerate(os.listdir(path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#
#                     if ratio == 1:
#                         low_image = image
#                     elif ratio == 2:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     elif ratio == 4:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     elif ratio == 8:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#
#                     zH = encoder(image.reshape(1, 64, 64, 1))
#                     z = encoder(low_image.reshape(1, 64, 64, 1))
#                     zH, z = tf.reshape(zH, [200]), tf.reshape(z, [200])
#                     database_image.append(low_image)
#                     database_gt.append(zH)
#                     database_latent.append(z)
#
#         database_latent, database_gt, database_image = np.array(database_latent), np.array(database_gt), np.array(database_image)
#         print(database_latent.shape, database_gt.shape, database_image.shape)
#         loss = [[0 for i in range(13)] for i in range(3)]
#
#         learning_rate, lr = [], 1
#         for i in range(10):
#             lr *= 0.8
#             learning_rate.append(lr)
#         learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 ,0.2, 0.1, 0.01, 0]
#         print(learning_rate)
#
#         syn = generator(tf.reshape(database_latent, [-1, 200]))
#         down_syn = down_image(syn, ratio)
#         syn_score = discriminator(syn)
#
#         if loss_type == '1':
#             #image loss
#             loss[0][0] += tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1)))
#             loss[1][0] += tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1)))
#             loss[2][0] += tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1)))
#             res_loss = tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1)))
#             # loss[0][0] += tf.reduce_mean(tf.square(database_gt - database_latent))
#             # loss[1][0] += tf.reduce_mean(tf.square(database_gt - database_latent))
#             # loss[2][0] += tf.reduce_mean(tf.square(database_gt - database_latent))
#             # res_loss = tf.reduce_mean(tf.square(database_gt - database_latent))
#         elif loss_type == '2':
#             #style loss
#             loss[0][0] += style_loss(database_image.reshape(-1, 64, 64, 1), down_syn)
#             loss[1][0] += style_loss(database_image.reshape(-1, 64, 64, 1), down_syn)
#             loss[2][0] += style_loss(database_image.reshape(-1, 64, 64, 1), down_syn)
#             res_loss = style_loss(database_image.reshape(-1, 64, 64, 1), down_syn)
#         elif loss_type == '3':
#             #adv loss
#             loss[0][0] += tf.reduce_mean(tf.square(syn_score - 1))
#             loss[1][0] += tf.reduce_mean(tf.square(syn_score - 1))
#             loss[2][0] += tf.reduce_mean(tf.square(syn_score - 1))
#             res_loss = tf.reduce_mean(tf.square(syn_score - 1))
#
#
#         zg = database_latent
#         z_final = zg
#         for t in range(3):
#             _, _, zreg = reg(zg)
#             dzreg = zreg - zg
#             for index, w in enumerate(learning_rate):
#                 z_output = zg + (w * dzreg)
#                 reg_syn = generator(z_output)
#                 reg_syn_score = discriminator(reg_syn)
#                 reg_down_syn = down_image(reg_syn, ratio)
#
#                 if loss_type == '1':
#                     if tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) < res_loss:
#                         res_loss = tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1)))
#                         z_final = zg + (w * dzreg)
#                     loss[t][index + 1] = tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1)))
#
#                 elif loss_type == '2':
#                     if style_loss(database_image.reshape(-1, 64, 64, 1), reg_down_syn) < res_loss:
#                         res_loss = style_loss(database_image.reshape(-1, 64, 64, 1), reg_down_syn)
#                         z_final = zg + (w * dzreg)
#                     loss[t][index + 1] = style_loss(database_image.reshape(-1, 64, 64, 1), reg_down_syn)
#
#                 elif loss_type == '3':
#                     if tf.reduce_mean(tf.square(reg_syn_score - 1)) < res_loss:
#                         res_loss = tf.reduce_mean(tf.square(reg_syn_score - 1))
#                         z_final = zg + (w * dzreg)
#                     loss[t][index + 1] = tf.reduce_mean(tf.square(reg_syn_score - 1))
#             zg = z_final
#
#         loss = np.array(loss)
#         x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
#         plt.plot(x, loss[0], marker='o')
#         plt.plot(x, loss[1], marker='o')
#         plt.plot(x, loss[2], marker='o')
#         plt.xlabel('W')
#         plt.ylabel('latent MSE')
#         plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')
#
#         for y in range(3):
#             for i, j in zip(x, loss[y]):
#                 plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
#         plt.show()
#         return loss, np.min(loss), np.max(loss)
#
#     loss1, min1_loss, max1_loss = search(loss_type='3', ratio=1)
#     loss2, min2_loss, max2_loss = search(loss_type='3', ratio=2)
#     loss3, min3_loss, max3_loss = search(loss_type='3', ratio=4)
#     loss4, min4_loss, max4_loss = search(loss_type='3', ratio=8)
#     print(f'1 ratio img loss is {loss1}, min {min1_loss}, max {max1_loss}')
#     print(f'2 ratio img loss is {loss2}, min {min2_loss}, max {max2_loss}')
#     print(f'4 ratio img loss is {loss3}, min {min3_loss}, max {max3_loss}')
#     print(f'8 ratio img loss is {loss4}, min {min4_loss}, max {max4_loss}')

def record_min_max_repeat_reg(train=False):
    global encoder
    global reg
    global generator
    global discriminator

    encoder = encoder()
    reg = regression()
    generator = generator()
    discriminator = discriminator()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg')
    generator.load_weights('weights/generator2')
    discriminator.load_weights('weights/discriminator2')

    def down_image(image, ratio):
        if ratio == 1:
            return tf.cast(image, dtype=tf.float32)
        if ratio == 2:
            down_syn = tf.image.resize(image, [32, 32], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 4:
            down_syn = tf.image.resize(image, [16, 16], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 8:
            down_syn = tf.image.resize(image, [8, 8], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn

    def style(real, fake):
        feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = feature_extraction(real)
        fake_feature = feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def distillation_loss(target, database_z, database_zreg):
        for latent in target:
            latent = tf.tile(tf.reshape(latent, [-1, 200]), [database_z.shape[0], 1])
            z_neighbor_distance = target - database_z

        return distance

    if train: path = '/disk2/bosen/Datasets/AR_train/'
    else: path = '/disk2/bosen/Datasets/AR_test/'

    def get_database():
        train_path = '/disk2/bosen/Datasets/AR_train/'
        test_path = '/disk2/bosen/Datasets/AR_test/'

        database_z, database_zreg = [], []
        for id_num, id in enumerate(os.listdir(train_path)):
            for file_num, filename in enumerate(os.listdir(train_path + id)):
                if 0 <= file_num < 5:
                    image = cv2.imread(train_path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    z = encoder(image.reshape(1, 64, 64, 1))
                    _, _, zreg = reg(z)
                    database_z.append(tf.reshape(z, [200]))
                    database_zreg.append(tf.reshape(zreg, [200]))

        for id_num, id in enumerate(os.listdir(test_path)):
            for file_num, filename in enumerate(os.listdir(test_path + id)):
                if 0 <= file_num < 5:
                    image = cv2.imread(test_path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    z = encoder(image.reshape(1, 64, 64, 1))
                    _, _, zreg = reg(z)
                    database_z.append(tf.reshape(z, [200]))
                    database_zreg.append(tf.reshape(zreg, [200]))

        database_z, database_zreg = np.array(database_z), np.array(database_zreg)

        dot_product_z_space = tf.matmul(database_z, tf.transpose(database_z))
        dot_product_zreg_space = tf.matmul(database_zreg, tf.transpose(database_zreg))
        square_norm_z_space = tf.linalg.diag_part(dot_product_z_space)
        square_norm_zreg_space = tf.linalg.diag_part(dot_product_zreg_space)

        distances_z = tf.sqrt(tf.expand_dims(square_norm_z_space, 1) - 2.0 * dot_product_z_space + tf.expand_dims(square_norm_z_space,0) + 1e-8) / 2
        distances_zreg = tf.sqrt(tf.expand_dims(square_norm_zreg_space, 1) - 2.0 * dot_product_zreg_space + tf.expand_dims(square_norm_zreg_space, 0) + 1e-8) / 2

        mean_distances_z = (tf.reduce_sum(distances_z)) / (math.factorial(distances_z.shape[0]) / (math.factorial(2) * math.factorial(int(distances_z.shape[0] - 2))))
        mean_distances_zreg = (tf.reduce_sum(distances_zreg)) / (math.factorial(distances_zreg.shape[0]) / (math.factorial(2) * math.factorial(int(distances_zreg.shape[0] - 2))))

        return database_z, database_zreg,  mean_distances_z, mean_distances_zreg


    def search(ratio):
        database_latent, database_gt, database_image = [], [], []
        for id_num, id in enumerate(os.listdir(path)):
            for file_num, filename in enumerate(os.listdir(path + id)):
                if 0 <= file_num < 5:
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                    if ratio == 1:
                        low_image = image
                    elif ratio == 2:
                        low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    elif ratio == 4:
                        low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    elif ratio == 8:
                        low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    zH = encoder(image.reshape(1, 64, 64, 1))
                    z = encoder(low_image.reshape(1, 64, 64, 1))
                    zH, z = tf.reshape(zH, [200]), tf.reshape(z, [200])
                    database_image.append(low_image)
                    database_gt.append(zH)
                    database_latent.append(z)

        database_latent, database_gt, database_image = np.array(database_latent), np.array(database_gt), np.array(database_image)
        print(database_latent.shape, database_gt.shape, database_image.shape)
        total_loss = [[0 for i in range(13)] for i in range(3)]
        image_loss = [[0 for i in range(13)] for i in range(3)]
        style_loss = [[0 for i in range(13)] for i in range(3)]
        adv_loss = [[0 for i in range(13)] for i in range(3)]
        latent_score = [[0 for i in range(13)] for i in range(3)]

        learning_rate, lr = [], 1
        for i in range(10):
            lr *= 0.8
            learning_rate.append(lr)
        learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 ,0.2, 0.1, 0.01, 0]
        print(learning_rate)

        syn = generator(tf.reshape(database_latent, [-1, 200]))
        down_syn = down_image(syn, ratio)
        syn_score = discriminator(syn)

        latent_score[0][0] = (tf.reduce_mean(tf.square(database_latent - database_gt)))
        latent_score[1][0] = (tf.reduce_mean(tf.square(database_latent - database_gt)))
        latent_score[2][0] = (tf.reduce_mean(tf.square(database_latent - database_gt)))

        image_loss[0][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))
        image_loss[1][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))
        image_loss[2][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))

        style_loss[0][0] = (style(database_image.reshape(-1, 64, 64, 1), down_syn))
        style_loss[1][0] = (style(database_image.reshape(-1, 64, 64, 1), down_syn))
        style_loss[2][0] = (style(database_image.reshape(-1, 64, 64, 1), down_syn))

        adv_loss[0][0] = (tf.reduce_mean(tf.square(syn_score - 1)))
        adv_loss[1][0] = (tf.reduce_mean(tf.square(syn_score - 1)))
        adv_loss[2][0] = (tf.reduce_mean(tf.square(syn_score - 1)))

        total_loss[0][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))
        total_loss[1][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))
        total_loss[2][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))
        res_loss = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))))
        print(res_loss)

        zg = database_latent
        z_final = zg
        for t in range(3):
            _, _, zreg = reg(zg)
            dzreg = zreg - zg

            for index, w in enumerate(learning_rate):
                z_output = zg + (w * dzreg)
                reg_syn = generator(z_output)
                reg_syn_score = discriminator(reg_syn)
                reg_down_syn = down_image(reg_syn, ratio)
                if (tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1)))) < res_loss:
                    res_loss = (tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))))
                    z_final = zg + (w * dzreg)

                total_loss[t][index + 1] = (tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))))
                image_loss[t][index + 1] = (tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))))
                style_loss[t][index + 1] = (style(database_image.reshape(-1, 64, 64, 1), reg_down_syn))
                adv_loss[t][index + 1] = (tf.reduce_mean(tf.square(reg_syn_score - 1)))
                latent_score[t][index + 1] = (tf.reduce_mean(tf.square(z_output - database_gt)))
            zg = z_final

        for loss in [[total_loss, 'image_loss'], [latent_score, 'latent error'], [image_loss, 'image_loss'], [style_loss, 'style_loss'], [adv_loss, 'adv_loss']]:
            loss_curve = np.array(loss[0])
            loss_name = loss[1]
            x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            plt.plot(x, loss_curve[0], marker='o')
            plt.plot(x, loss_curve[1], marker='o')
            plt.plot(x, loss_curve[2], marker='o')
            plt.title(loss_name)
            plt.xlabel('W')
            plt.ylabel('Error')
            plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')

            for y in range(3):
                for i, j in zip(x, loss_curve[y]):
                    plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
            plt.show()

        return total_loss, latent_score, image_loss, style_loss, adv_loss

    database_z, database_zreg, mean_z, mean_zreg = get_database()
    print(mean_z, mean_zreg)
    # _, _, image1_loss, style1_loss, adv1_loss = search(ratio = 2)
    # _, _, image2_loss, style2_loss, adv2_loss = search(ratio = 4)
    # _, _, image3_loss, style3_loss, adv3_loss = search(ratio = 8)
    #
    # print(np.min(image1_loss), np.max(image1_loss))
    # print(np.min(style1_loss), np.max(style1_loss))
    # print(np.min(adv1_loss), np.max(adv1_loss))
    #
    # print(np.min(image2_loss), np.max(image2_loss))
    # print(np.min(style2_loss), np.max(style2_loss))
    # print(np.min(adv2_loss), np.max(adv2_loss))
    #
    # print(np.min(image3_loss), np.max(image3_loss))
    # print(np.min(style3_loss), np.max(style3_loss))
    # print(np.min(adv3_loss), np.max(adv3_loss))


# def record_min_max_repeat_reg(train=False):
#     global encoder
#     global reg
#     global generator
#     global discriminator
#
#     encoder = encoder()
#     reg = regression()
#     generator = generator()
#     discriminator = discriminator()
#
#     encoder.load_weights('weights/encoder')
#     reg.load_weights('weights/reg')
#     generator.load_weights('weights/generator2')
#     discriminator.load_weights('weights/discriminator2')
#
#     def distillation_loss(target_z, target_zreg, database_z, database_zreg, mean_z, mean_zreg):
#         dis_loss = 0
#         for num, (latent_z, latent_zreg) in enumerate(zip(target_z, target_zreg)):
#             latent_z_expand = tf.tile(tf.reshape(latent_z, [-1, 200]), [database_z.shape[0], 1])
#
#             z_neighbor_distance = tf.sqrt(tf.reduce_sum(tf.square(latent_z_expand - database_z), axis=-1))
#             z_neighbor_distance = z_neighbor_distance.numpy()
#
#             for i in range(3):
#                 min_index = np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]
#                 dis_loss += tf.math.abs(((tf.sqrt(tf.reduce_sum(tf.square(latent_z - database_z[min_index])))/ mean_z) - (tf.sqrt(tf.reduce_sum(tf.square(latent_zreg - database_zreg[min_index])))) / mean_zreg))
#                 z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
#
#         return dis_loss / (num + 1)
#
#
#     if train:
#         path = '/disk2/bosen/Datasets/AR_train/'
#     else:
#         path = '/disk2/bosen/Datasets/AR_test/'
#
#     def get_database():
#         train_path = '/disk2/bosen/Datasets/AR_train/'
#         test_path = '/disk2/bosen/Datasets/AR_test/'
#
#         database_z, database_zreg = [], []
#         for id_num, id in enumerate(os.listdir(train_path)):
#             for file_num, filename in enumerate(os.listdir(train_path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(train_path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     z = encoder(image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     database_z.append(tf.reshape(z, [200]))
#                     database_zreg.append(tf.reshape(zreg, [200]))
#
#         for id_num, id in enumerate(os.listdir(test_path)):
#             for file_num, filename in enumerate(os.listdir(test_path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(test_path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     z = encoder(image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     database_z.append(tf.reshape(z, [200]))
#                     database_zreg.append(tf.reshape(zreg, [200]))
#
#         database_z, database_zreg = np.array(database_z), np.array(database_zreg)
#
#         dot_product_z_space = tf.matmul(database_z, tf.transpose(database_z))
#         dot_product_zreg_space = tf.matmul(database_zreg, tf.transpose(database_zreg))
#         square_norm_z_space = tf.linalg.diag_part(dot_product_z_space)
#         square_norm_zreg_space = tf.linalg.diag_part(dot_product_zreg_space)
#
#         distances_z = tf.sqrt(tf.expand_dims(square_norm_z_space, 1) - 2.0 * dot_product_z_space + tf.expand_dims(square_norm_z_space, 0) + 1e-8) / 2
#         distances_zreg = tf.sqrt(tf.expand_dims(square_norm_zreg_space, 1) - 2.0 * dot_product_zreg_space + tf.expand_dims(square_norm_zreg_space, 0) + 1e-8) / 2
#
#         mean_distances_z = (tf.reduce_sum(distances_z)) / (math.factorial(distances_z.shape[0]) / (math.factorial(2) * math.factorial(int(distances_z.shape[0] - 2))))
#         mean_distances_zreg = (tf.reduce_sum(distances_zreg)) / (math.factorial(distances_zreg.shape[0]) / (math.factorial(2) * math.factorial(int(distances_zreg.shape[0] - 2))))
#
#         return database_z, database_zreg, mean_distances_z, mean_distances_zreg
#
#     def search(ratio):
#         database_latent_z, database_latent_zreg, database_gt, database_image = [], [], [], []
#
#         for id_num, id in enumerate(os.listdir(path)):
#             for file_num, filename in enumerate(os.listdir(path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#
#                     if ratio == 1:
#                         low_image = image
#                     elif ratio == 2:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     elif ratio == 4:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     elif ratio == 8:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#
#                     zH = encoder(image.reshape(1, 64, 64, 1))
#                     z = encoder(low_image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     zH, z, zreg = tf.reshape(zH, [200]), tf.reshape(z, [200]), tf.reshape(zreg, [200])
#                     database_image.append(low_image)
#                     database_gt.append(zH)
#                     database_latent_z.append(z)
#                     database_latent_zreg.append(zreg)
#
#
#         database_latent_z, database_latent_zreg, database_gt, database_image = np.array(database_latent_z), np.array(database_latent_zreg), np.array(database_gt), np.array(database_image)
#         database_z, database_zreg, mean_z, mean_zreg = get_database()
#
#         dis_loss = [[0 for i in range(13)] for i in range(3)]
#
#         learning_rate, lr = [], 1
#         for i in range(10):
#             lr *= 0.8
#             learning_rate.append(lr)
#         learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0]
#         print(learning_rate)
#
#
#         dis_loss[0][0] = distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg)
#         dis_loss[1][0] = distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg)
#         dis_loss[2][0] = distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg)
#
#
#         res_loss = distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg)
#         print(distillation_loss(database_latent_z, database_latent_z, database_z, database_zreg, mean_z, mean_zreg))
#         print(res_loss)
#
#         zg = database_latent_z
#         z_final = zg
#         for t in range(3):
#             _, _, zreg = reg(zg)
#             dzreg = zreg - zg
#
#             for index, w in enumerate(learning_rate):
#                 z_output = zg + (w * dzreg)
#                 if distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) < res_loss:
#                     res_loss = distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg)
#                     z_final = zg + (w * dzreg)
#
#                 dis_loss[t][index + 1] = distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg)
#             zg = z_final
#
#         for loss in [[dis_loss, 'dis_loss']]:
#             loss_curve = np.array(loss[0])
#             loss_name = loss[1]
#             x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
#             plt.plot(x, loss_curve[0], marker='o')
#             plt.plot(x, loss_curve[1], marker='o')
#             plt.plot(x, loss_curve[2], marker='o')
#             plt.title(loss_name)
#             plt.xlabel('W')
#             plt.ylabel('Error')
#             plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')
#
#             for y in range(3):
#                 for i, j in zip(x, loss_curve[y]):
#                     plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
#             plt.show()
#
#         return dis_loss
#
#     database_z, database_zreg, mean_z, mean_zreg = get_database()
#     print(mean_z, mean_zreg)
#     dis1_loss = search(ratio = 2)
#     dis2_loss = search(ratio = 4)
#     dis3_loss = search(ratio = 8)
#
#     print(np.min(dis1_loss), np.max(dis1_loss))
#     print(np.min(dis2_loss), np.max(dis2_loss))
#     print(np.min(dis3_loss), np.max(dis3_loss))

def PCA_before_x_after_reg():
    global encoder
    global reg
    encoder = encoder()
    reg = regression()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg_x_cls_REG')

    train_path = '/disk2/bosen/Datasets/AR_train/'
    test_path = '/disk2/bosen/Datasets/AR_aligment_other/'

    pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio = [], [[] for i in range(3)], [[] for i in range(3)], [[]for i in range(3)], [[] for i in range(3)]
    for id in os.listdir(train_path):
        print(int(id[2:]), end=',')
        for file_num, filename in enumerate(os.listdir(train_path + id)):
            if file_num == 10:
                break
            image = cv2.imread(train_path + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            z, z1, z2, z3 = encoder(image.reshape(1, 64, 64, 1)), encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
            _, _, z = reg(z)
            _, _, z1 = reg(z1)
            _, _, z2 = reg(z2)
            _, _, z3 = reg(z3)

            pca_data.append(tf.reshape(z, [200]))
            pca_data.append(tf.reshape(z1, [200]))
            pca_data.append(tf.reshape(z2, [200]))
            pca_data.append(tf.reshape(z3, [200]))

            # if (int(id[2:]) == 1):
            #     pca_1ratio[0].append(tf.reshape(z, [200]))
            #     pca_2ratio[0].append(tf.reshape(z1, [200]))
            #     pca_4ratio[0].append(tf.reshape(z2, [200]))
            #     pca_8ratio[0].append(tf.reshape(z3, [200]))
            # elif (int(id[2:]) == 2):
            #     pca_1ratio[1].append(tf.reshape(z, [200]))
            #     pca_2ratio[1].append(tf.reshape(z1, [200]))
            #     pca_4ratio[1].append(tf.reshape(z2, [200]))
            #     pca_8ratio[1].append(tf.reshape(z3, [200]))
            # elif (int(id[2:]) == 4):
            #     pca_1ratio[2].append(tf.reshape(z, [200]))
            #     pca_2ratio[2].append(tf.reshape(z1, [200]))
            #     pca_4ratio[2].append(tf.reshape(z2, [200]))
            #     pca_8ratio[2].append(tf.reshape(z3, [200]))

    # pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio = np.array(pca_data), np.array(pca_1ratio), np.array(pca_2ratio), np.array(pca_4ratio), np.array(pca_8ratio)
    # print(pca_data.shape, pca_1ratio.shape, pca_2ratio.shape, pca_4ratio.shape, pca_8ratio.shape)

    for id in os.listdir(test_path):
        for file_num, filename in enumerate(os.listdir(test_path + id)):
            if (int(id[2:])!=85) and (int(id[2:])!=83) and (int(id[2:])!=84):
                continue

            if file_num == 5:
                break
            image = cv2.imread(test_path + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            z, z1, z2, z3 = encoder(image.reshape(1, 64, 64, 1)), encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
            _, _, z = reg(z)
            _, _, z1 = reg(z1)
            _, _, z2 = reg(z2)
            _, _, z3 = reg(z3)

            if (int(id[2:]) == 85):
                pca_1ratio[0].append(tf.reshape(z, [200]))
                pca_2ratio[0].append(tf.reshape(z1, [200]))
                pca_4ratio[0].append(tf.reshape(z2, [200]))
                pca_8ratio[0].append(tf.reshape(z3, [200]))
            elif (int(id[2:]) == 83):
                pca_1ratio[1].append(tf.reshape(z, [200]))
                pca_2ratio[1].append(tf.reshape(z1, [200]))
                pca_4ratio[1].append(tf.reshape(z2, [200]))
                pca_8ratio[1].append(tf.reshape(z3, [200]))
            elif (int(id[2:]) == 84):
                pca_1ratio[2].append(tf.reshape(z, [200]))
                pca_2ratio[2].append(tf.reshape(z1, [200]))
                pca_4ratio[2].append(tf.reshape(z2, [200]))
                pca_8ratio[2].append(tf.reshape(z3, [200]))

    pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio = np.array(pca_data), np.array(pca_1ratio), np.array(pca_2ratio), np.array(pca_4ratio), np.array(pca_8ratio)
    print(pca_data.shape, pca_1ratio.shape, pca_2ratio.shape, pca_4ratio.shape, pca_8ratio.shape)

    def visualize_pca(data, ratio1, ratio2, ratio4, ratio8):
        # 建立 PCA 模型，設定要保留的主成分數量
        pca = PCA(n_components=2)

        # 訓練 PCA 模型
        pca_result = pca.fit_transform(data)

        # 繪製 PCA 投影
        plt.figure(figsize=(10, 6))

        # 繪製灰色點表示原始資料
        plt.scatter(pca_result[:, 0], pca_result[:, 1], color='grey', alpha=0.5, label='Original Data')

        # 繪製不同顏色的點表示額外的向量投影
        for i, (data1, data2, data3, data4) in enumerate(zip(ratio1, ratio2, ratio4, ratio8)):
            data1_result = pca.transform(data1.reshape(-1, 200))
            data2_result = pca.transform(data2.reshape(-1, 200))
            data3_result = pca.transform(data3.reshape(-1, 200))
            data4_result = pca.transform(data4.reshape(-1, 200))

            if i == 0:
                color = 'blue'
            elif i == 1:
                color = 'red'
            elif i == 2:
                color = 'black'
            plt.scatter(data1_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 1 Ratio', marker='o', color=color)
            plt.scatter(data2_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 2 Ratio', marker='x', color=color)
            plt.scatter(data3_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 4 Ratio', marker='s', color=color)
            plt.scatter(data4_result[:, 0], data1_result[:, 1], label=f'ID {i + 1} 8 Ratio', marker='+', color=color)

        # 添加標題和標籤
        plt.title('PCA Projection of Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # 顯示圖例
        plt.legend()

        # 顯示圖形
        plt.show()

    # 視覺化 PCA 投影
    visualize_pca(pca_data, pca_1ratio, pca_2ratio, pca_4ratio, pca_8ratio)


def knn_test():
    global encoder
    global reg
    encoder = encoder()
    reg = regression()
    cls_z = cls()
    cls_zreg = cls()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg_x_cls_REG')
    cls_z.load_weights('weights/cls_z')
    cls_zreg.load_weights('weights/reg_x_cls_CLS')

    path_train = '/disk2/bosen/Datasets/AR_train/'
    path_test = '/disk2/bosen/Datasets/AR_aligment_other/'

    database_feature, database_label = [], []
    feature1ratio = []
    feature1ratio_label = []

    # for id in os.listdir(path_train):
    #     print(id)
    #     for num, filename in enumerate(os.listdir(path_train + id)):
    #         if 0 <= num < 10:
    #             image = cv2.imread(path_train + id + '/' + filename, 0) / 255
    #             image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
    #             blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
    #             low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #             low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #             low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    #
    #             feature = encoder(image.reshape(1, 64, 64, 1))
    #             feature1 = encoder(low1_image.reshape(1, 64, 64, 1))
    #             feature2 = encoder(low2_image.reshape(1, 64, 64, 1))
    #             feature3 = encoder(low3_image.reshape(1, 64, 64, 1))
    #
    #             _, _, feature = reg(feature)
    #             _, _, feature1 = reg(feature1)
    #             _, _, feature2 = reg(feature2)
    #             _, _, feature3 = reg(feature3)
    #
    #
    #             feature = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
    #             feature1 = feature1 / tf.sqrt(tf.reduce_sum(tf.square(feature1)))
    #             feature2 = feature2 / tf.sqrt(tf.reduce_sum(tf.square(feature2)))
    #             feature3 = feature3 / tf.sqrt(tf.reduce_sum(tf.square(feature3)))
    #
    #             database_feature.append(tf.reshape(feature, [200]))
    #             database_feature.append(tf.reshape(feature1, [200]))
    #             database_feature.append(tf.reshape(feature2, [200]))
    #             database_feature.append(tf.reshape(feature3, [200]))
    #
    #             database_label.append(int(id[2:]))
    #             database_label.append(int(id[2:]))
    #             database_label.append(int(id[2:]))
    #             database_label.append(int(id[2:]))

    # for ratio in os.listdir(path_test):
    #     for id in os.listdir(path_test + ratio):
    #         for num, filename in enumerate(os.listdir(path_test + ratio + '/' + id)):
    #             latent = np.load(path_test + ratio + '/' + id + '/' + filename)
    #             latent = latent / tf.sqrt(tf.reduce_sum(tf.square(latent)))
    #
    #             if ratio[0] == '2':
    #                 feature2ratio.append(tf.reshape(latent, [200]))
    #                 feature2ratio_label.append(int(id[2:]))
    #             if ratio[0] == '4':
    #                 feature4ratio.append(tf.reshape(latent, [200]))
    #                 feature4ratio_label.append(int(id[2:]))
    #             if ratio[0] == '8':
    #                 feature8ratio.append(tf.reshape(latent, [200]))
    #                 feature8ratio_label.append(int(id[2:]))

    num_z, num_zreg = 0, 0
    count = 0
    for id in os.listdir(path_test):
        for num, filename in enumerate(os.listdir(path_test + id)):
            count += 1
            image = cv2.imread(path_test + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            # low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64 ,64), cv2.INTER_CUBIC)

            feature_z = encoder(image.reshape(1, 64, 64, 1))
            _, _, feature_zreg = reg(feature_z)
            _, pred_z = cls_z(feature_z)
            _, pred_zreg = cls_zreg(feature_zreg)
            if int(id[2:])-1 == np.argmax(pred_z, axis=-1)[0]: num_z += 1
            if int(id[2:])-1 == np.argmax(pred_zreg, axis=-1)[0]: num_zreg += 1
    print(count)
    print(f'Accuracy z is {num_z / count}')
    print(f'Accuracy zreg is {num_zreg / count}')


            # feature = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
            # feature1ratio.append(tf.reshape(feature, [200]))
            # feature1ratio_label.append(int(id[2:]))





    # database_feature, database_label = np.array(database_feature), np.array(database_label)
    # feature1ratio = np.array(feature1ratio)
    # feature1ratio_label = np.array(feature1ratio_label)
    #
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(database_feature, database_label)
    # feature1ratio_pred = knn.predict(feature1ratio)
    #
    #
    # feature1ratio_cm = confusion_matrix(feature1ratio_label, feature1ratio_pred)
    # feature1ratio_acc = accuracy_score(feature1ratio_label, feature1ratio_pred)
    #
    # plt.title(f'1 Ratio Acc {str(feature1ratio_acc)[0:5]}')
    # plt.imshow(feature1ratio_cm, cmap='jet')
    # plt.show()


def reg_knn_cls_test(ratio=2):
    global encoder
    global reg
    global generator
    encoder = encoder()
    reg = regression()
    generator = generator()
    cls_z = cls()
    cls_zreg = cls()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg_x_cls_REG')
    generator.load_weights('weights/generator2')
    cls_z.load_weights('weights/cls_z')
    cls_zreg.load_weights('weights/reg_x_cls_CLS')

    path_train = '/disk2/bosen/Datasets/AR_train/'
    path_test = '/disk2/bosen/Datasets/AR_aligment_other/'

    database_feature, database_label = [], []

    for id in os.listdir(path_train):
        print(id)
        for num, filename in enumerate(os.listdir(path_train + id)):
            if 0 <= num < 10:
                image = cv2.imread(path_train + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                feature = encoder(image.reshape(1, 64, 64, 1))
                feature1 = encoder(low1_image.reshape(1, 64, 64, 1))
                feature2 = encoder(low2_image.reshape(1, 64, 64, 1))
                feature3 = encoder(low3_image.reshape(1, 64, 64, 1))

                _, _, feature = reg(feature)
                _, _, feature1 = reg(feature1)
                _, _, feature2 = reg(feature2)
                _, _, feature3 = reg(feature3)


                feature = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
                feature1 = feature1 / tf.sqrt(tf.reduce_sum(tf.square(feature1)))
                feature2 = feature2 / tf.sqrt(tf.reduce_sum(tf.square(feature2)))
                feature3 = feature3 / tf.sqrt(tf.reduce_sum(tf.square(feature3)))

                database_feature.append(tf.reshape(feature, [200]))
                database_feature.append(tf.reshape(feature1, [200]))
                database_feature.append(tf.reshape(feature2, [200]))
                database_feature.append(tf.reshape(feature3, [200]))

                database_label.append(int(id[2:])-1)
                database_label.append(int(id[2:])-1)
                database_label.append(int(id[2:])-1)
                database_label.append(int(id[2:])-1)

    database_feature, database_label = np.array(database_feature), np.array(database_label)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(database_feature, database_label)


    mPSNR = [[] for i in range(10)]
    mKNN = [0 for i in range(10)]
    mCLS = [0 for i in range(10)]
    image_number = 0
    for id in os.listdir(path_test):
        for num, filename in enumerate(os.listdir(path_test + id)):
            image_number += 1
            print(image_number)
            high_image = cv2.imread(path_test + id + '/' + filename, 0) / 255
            high_image = cv2.resize(high_image, (64, 64), cv2.INTER_CUBIC)
            if ratio == 2: image = cv2.resize(cv2.resize(high_image, (32, 32), cv2.INTER_CUBIC), (64 ,64), cv2.INTER_CUBIC)
            if ratio == 4: image = cv2.resize(cv2.resize(high_image, (16, 16), cv2.INTER_CUBIC), (64 ,64), cv2.INTER_CUBIC)
            if ratio == 8: image = cv2.resize(cv2.resize(high_image, (8, 8), cv2.INTER_CUBIC), (64 ,64), cv2.INTER_CUBIC)

            for step in range(0, 10):
                feature = encoder(tf.reshape(image, [1, 64, 64, 1]))
                _, _, feature = reg(feature)
                feature_knn = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))

                pred_knn = knn.predict(feature_knn)
                _, pred_cls = cls_zreg(feature)
                image = generator(feature)
                mPSNR[step].append(tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(image, dtype=tf.float32), max_val=1)[0])

                if int(id[2:]) - 1 == pred_knn: mKNN[step] += 1
                if int(id[2:]) - 1 == np.argmax(pred_cls, axis=-1)[0]: mCLS[step] += 1

    print(image_number)
    mPSNR, mKNN, mCLS = tf.reduce_mean(np.array(mPSNR), axis=-1), np.array(mKNN), np.array(mCLS)
    mKNN, mCLS = (mKNN/image_number), (mCLS/image_number)
    print(mPSNR.shape, mKNN.shape, mCLS.shape)
    print(mPSNR, mKNN, mCLS)

    plt.plot(mPSNR, marker='o', label='mPSNR')
    plt.grid(True)
    plt.title(f'{ratio} ratio mPSNR')
    plt.xlabel('iterate')
    plt.ylabel('number')
    plt.legend()
    plt.show()

    plt.plot(mKNN, marker='o', label='mKNN')
    plt.grid(True)
    plt.title(f'{ratio} ratio mKNN')
    plt.xlabel('iterate')
    plt.ylabel('number')
    plt.legend()
    plt.show()

    plt.plot(mCLS, marker='o', label='mCLS')
    plt.grid(True)
    plt.title(f'{ratio} ratio mCLS')
    plt.xlabel('iterate')
    plt.ylabel('number')
    plt.legend()
    plt.show()





    # mPSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])


def ar_other_data_resolution_augmentation():
    path = '/disk2/bosen/Datasets/AR_aligment_other/'
    target_path = '/disk2/bosen/Datasets/AR_aligment_other_blurkernel7/'
    ID = [f'ID0{i}' if i < 10 else f'ID{i}' for i in range(1, 91)]
    for id in ID:
        for num, filename in enumerate(os.listdir(path + id)):
            if num == 1:
                break
            image = cv2.imread(path + id + '/' + filename, 0) / 255
            image= cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            plt.imshow(blur_gray, cmap='gray')
            plt.show()

            # cv2.imwrite(target_path + f'ID{int(id[2:])}_{filename[0:-4]}_2ratio.jpg', low1_image*255)
            # cv2.imwrite(target_path + f'ID{int(id[2:])}_{filename[0:-4]}_4ratio.jpg', low2_image*255)
            # cv2.imwrite(target_path + f'ID{int(id[2:])}_{filename[0:-4]}_8ratio.jpg', low3_image*255)


def ar_other_data_aligment_augmentation():

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/bosen/PycharmProjects/Datasets/shape_predictor_68_face_landmarks.dat")

    read_path = '/disk2/bosen/Datasets/AR_aligment_other/'
    affine_path = f'/disk2/bosen/Datasets/AR_original/'

    total_affine_path = []
    for file in os.listdir(affine_path):
        for filename in os.listdir(affine_path + file):
            total_affine_path.append(affine_path + file + '/' + filename)

    get_read_path = os.listdir(read_path)
    get_read_path = sorted(get_read_path, key=natural_sort_key)
    print(total_affine_path)


    count = 0
    for id_num, id in enumerate(get_read_path):
        for num, filename in enumerate(os.listdir(read_path + id)):
            print(total_affine_path.index(+filename))
            print(filename)
            # print(read_path + get_read_path[id_num] + '/' + filename)
            # read_img = cv2.imread(read_path + get_read_path[id_num] + '/' + filename, cv2.IMREAD_GRAYSCALE)


            # x1 = 61
            # y1 = 76
            # x2 = 43
            # y2 = 40
            # x3 = 81
            # y3 = 41
            # rects = detector(img, 0)
            # z1 = []
            # z2 = []
            # if len(rects) != 0:
            #     count += 1
            # for i in range(len(rects)):
            #     landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
            #     xx1 = landmarks[33][0, 0]
            #     yy1 = landmarks[33][0, 1]
            #     xx2 = landmarks[39][0, 0]
            #     yy2 = landmarks[39][0, 1]
            #     xx3 = landmarks[42][0, 0]
            #     yy3 = landmarks[42][0, 1]
            #     print(xx1, yy1, xx2, yy2, xx3, yy3)
            #
            #     pts1 = np.float32([[xx1, yy1], [xx2, yy2], [xx3, yy3]])
            #     pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
            #     M = cv2.getAffineTransform(pts1, pts2)
            #     res = cv2.warpAffine(img, M, (128, 128))
            #     # plt.scatter(pts2[:, 0:1], pts2[:, 1:2], color='r')
            #     # plt.imshow(res, cmap='gray')
            #     # plt.axis('off')
            #     # plt.show()
            #     # res = cv2.equalizeHist(res)
            #     # plt.imshow(res, cmap='gray')
            #     # plt.axis('off')
            #     # plt.show()
            #     # cv2.imwrite(f"AR_aligment_initial/dbf1/{filename}", res)
            #     plt.imshow(res, cmap='gray')
            #     plt.show()
            #
            # #     for idx, point in enumerate(landmarks):
            # #         z1.append(point[0, 0])
            # #         z2.append(point[0, 1])
            # # re = cv2.resize(img[min(z2)-5:max(z2)-5, min(z1):max(z1)], (64, 64), interpolation=cv2.INTER_CUBIC)
            # print(count)


if __name__ == '__main__':
    pass
    # record_min_max_repeat_reg(train=False)
    # PCA_before_x_after_reg()
    # reg_knn_cls_test(ratio=2)
    # ar_other_data_resolution_augmentation()
    # ar_other_data_aligment_augmentation()





