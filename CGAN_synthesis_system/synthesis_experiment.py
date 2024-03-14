from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
from build_model import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


def zd_zg_space():
    def get_feature(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        z1, _ = encoder(low1_image)
        z2, _ = encoder(low2_image)
        z3, _ = encoder(low3_image)
        zdh, _ = ztozd(z1)
        zdm, _ = ztozd(z2)
        zdl, _ = ztozd(z3)
        zgh, _, _ = ztozg(z1)
        zgm, _, _ = ztozg(z2)
        zgl, _, _ = ztozg(z3)
        return tf.reshape(zdh, [200]), tf.reshape(zdm, [200]), tf.reshape(zdl, [200]), tf.reshape(zgh, [200]), tf.reshape(zgm, [200]), tf.reshape(zgl, [200])

    def downscale(image):
        H = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        return H, low1_image, low2_image, low3_image

    global encoder
    global ztozd
    global ztozg

    encoder = normal_encoder()
    ztozd = ZtoZd()
    ztozg = ZtoZg()

    encoder.load_weights('model_weight/AE_encoder')
    ztozd.load_weights('model_weight/AE_ztozd')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')

    path_AR = '/home/bosen/PycharmProjects/Datasets/AR_train/'
    zgH_feature, zgh_feature, zgm_feature, zgl_feature, zg_feature = [], [], [], [], []
    zgH_feature_target, zgh_feature_target, zgm_feature_target, zgl_feature_target, zg_feature_target = [], [], [], [], []


    # zdh_feature, zdm_feature, zdl_feature, zgh_feature, zgm_feature, zgl_feature = [], [], [], [], [], []
    # zd_feature, zg_feature = [], []
    images = []
    ID = [f"ID{i}" for i in range(1, 91)]
    for id in ID:
        for num, filename in enumerate(os.listdir(path_AR + id)):
            if num < 20:
                image = cv2.imread(path_AR + id + '/' + filename, 0) / 255
                H, low1, low2, low3 = downscale(image)
                z1, z2, z3, z4 = encoder(H.reshape(1, 64, 64, 1)), encoder(low1.reshape(1, 64, 64, 1)), encoder(low2.reshape(1, 64, 64, 1)), encoder(low3.reshape(1, 64, 64, 1))
                zg1, _, _ = ztozg(z1)
                zg2, _, _ = ztozg(z2)
                zg3, _, _ = ztozg(z3)
                zg4, _, _ = ztozg(z4)

                if id == f'ID77':
                    zgH_feature_target.append(tf.reshape(zg1, [200]))
                    zgh_feature_target.append(tf.reshape(zg2, [200]))
                    zgm_feature_target.append(tf.reshape(zg3, [200]))
                    zgl_feature_target.append(tf.reshape(zg4, [200]))

                zgH_feature.append(tf.reshape(zg1, [200]))
                zgh_feature.append(tf.reshape(zg2, [200]))
                zgm_feature.append(tf.reshape(zg3, [200]))
                zgl_feature.append(tf.reshape(zg4, [200]))

                zg_feature.append(tf.reshape(zg1, [200]))
                zg_feature.append(tf.reshape(zg2, [200]))
                zg_feature.append(tf.reshape(zg3, [200]))
                zg_feature.append(tf.reshape(zg4, [200]))

                # images.append(low1), images.append(low2), images.append(low3)
                # zdh, zdm, zdl, zgh, zgm, zgl = get_feature(image)
                # zdh_feature.append(zdh), zdm_feature.append(zdm), zdl_feature.append(zdl)
                # zgh_feature.append(zgh), zgm_feature.append(zgm), zgl_feature.append(zgl)
                # zd_feature.append(zdh), zd_feature.append(zdm), zd_feature.append(zdl)
                # zg_feature.append(zgh), zg_feature.append(zgm), zg_feature.append(zgl)

    zg_feature, zgH_feature, zgh_feature, zgm_feature, zgl_feature = np.array(zg_feature), np.array(zgH_feature), np.array(zgh_feature), np.array(zgm_feature), np.array(zgl_feature)
    print(zg_feature.shape, zgH_feature.shape, zgh_feature.shape, zgm_feature.shape, zgl_feature.shape)
    pca = PCA(n_components=2)
    pca.fit_transform(zg_feature)
    zgH_feature = pca.transform(zgH_feature_target)
    zgh_feature = pca.transform(zgh_feature_target)
    zgm_feature = pca.transform(zgm_feature_target)
    zgl_feature = pca.transform(zgl_feature_target)

    plt.figure(figsize=(7, 5))
    plt.style.use('ggplot')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Zg cross resolution distribution')
    plt.scatter(zgH_feature[:][:, 0], zgH_feature[:][:, 1], c='b', marker='o', s=25)
    plt.scatter(zgh_feature[:][:, 0], zgh_feature[:][:, 1], c='g', marker='o', s=25)
    plt.scatter(zgm_feature[:][:, 0], zgm_feature[:][:, 1], c='y', marker='o', s=25)
    plt.scatter(zgl_feature[:][:, 0], zgl_feature[:][:, 1], c='r', marker='o', s=25)

    # plt.scatter(zgh_feature[60:][:, 0], zgh_feature[60:][:, 1], c='g', marker='o', s=25)
    # plt.scatter(zgm_feature[60:][:, 0], zgm_feature[60:][:, 1], c='k', marker='o', s=25)
    # plt.scatter(zgl_feature[60:][:, 0], zgl_feature[60:][:, 1], c='r', marker='o', s=25)
    #
    # plt.scatter(zgh_feature[0:20][:, 0], zgh_feature[0:20][:, 1], c='g', marker='*', s=250)
    # plt.scatter(zgh_feature[20:40][:, 0], zgh_feature[20:40][:, 1], c='g', marker=6, s=250)
    # plt.scatter(zgh_feature[40:60][:, 0], zgh_feature[40:60][:, 1], c='g', marker=7, s=250)
    #
    # plt.scatter(zgm_feature[0:20][:, 0], zgm_feature[0:20][:, 1], c='k', marker='*', s=250)
    # plt.scatter(zgm_feature[20:40][:, 0], zgm_feature[20:40][:, 1], c='k', marker=6, s=250)
    # plt.scatter(zgm_feature[40:60][:, 0], zgm_feature[40:60][:, 1], c='k', marker=7, s=250)
    #
    # plt.scatter(zgl_feature[0:20][:, 0], zgl_feature[0:20][:, 1], c='r', marker='*', s=250)
    # plt.scatter(zgl_feature[20:40][:, 0], zgl_feature[20:40][:, 1], c='r', marker=6, s=250)
    # plt.scatter(zgl_feature[40:60][:, 0], zgl_feature[40:60][:, 1], c='r', marker=7, s=250)
    plt.legend(['Real H', '2-ratio resolution', '4-ratio resolution', '8-ratio resolution'], loc='upper left')
    plt.show()
    plt.close()




    # pca = PCA(n_components=2)
    # pca.fit_transform(zd_feature)
    # zdh_feature = pca.transform(zdh_feature)
    # zdm_feature = pca.transform(zdm_feature)
    # zdl_feature = pca.transform(zdl_feature)
    # plt.figure(figsize=(7, 5))
    # plt.style.use('ggplot')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    # plt.title('Zd cross resolution distribution')
    # plt.scatter(zdh_feature[60:][:, 0], zdh_feature[60:][:, 1], c='g', marker='o', s=25)
    # plt.scatter(zdm_feature[60:][:, 0], zdm_feature[60:][:, 1], c='k', marker='o', s=25)
    # plt.scatter(zdl_feature[60:][:, 0], zdl_feature[60:][:, 1], c='r', marker='o', s=25)
    #
    # plt.scatter(zdh_feature[0:20][:, 0], zdh_feature[0:20][:, 1], c='g', marker='*', s=250)
    # plt.scatter(zdh_feature[20:40][:, 0], zdh_feature[20:40][:, 1], c='g', marker=6, s=250)
    # plt.scatter(zdh_feature[40:60][:, 0], zdh_feature[40:60][:, 1], c='g', marker=7, s=250)
    #
    # plt.scatter(zdm_feature[0:20][:, 0], zdm_feature[0:20][:, 1], c='k', marker='*', s=250)
    # plt.scatter(zdm_feature[20:40][:, 0], zdm_feature[20:40][:, 1], c='k', marker=6, s=250)
    # plt.scatter(zdm_feature[40:60][:, 0], zdm_feature[40:60][:, 1], c='k', marker=7, s=250)
    #
    # plt.scatter(zdl_feature[0:20][:, 0], zdl_feature[0:20][:, 1], c='r', marker='*', s=250)
    # plt.scatter(zdl_feature[20:40][:, 0], zdl_feature[20:40][:, 1], c='r', marker=6, s=250)
    # plt.scatter(zdl_feature[40:60][:, 0], zdl_feature[40:60][:, 1], c='r', marker=7, s=250)
    # plt.legend(['2_ratio zd latent code', '4-ratio zd latent code', '8-ratio zd latent code'], loc='upper left')
    # plt.show()
    # plt.close()

def zd_zg_space_find_hard_sample():
    def get_feature(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        z1, _ = encoder(low1_image)
        z2, _ = encoder(low2_image)
        z3, _ = encoder(low3_image)
        zdh, _ = ztozd(z1)
        zdm, _ = ztozd(z2)
        zdl, _ = ztozd(z3)
        zgh, _, _ = ztozg(z1)
        zgm, _, _ = ztozg(z2)
        zgl, _, _ = ztozg(z3)
        return tf.reshape(zdh, [200]), tf.reshape(zdm, [200]), tf.reshape(zdl, [200]), tf.reshape(zgh, [200]), tf.reshape(zgm, [200]), tf.reshape(zgl, [200])

    def downscale(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        return low1_image, low2_image, low3_image

    global encoder
    global ztozd
    global ztozg

    encoder = encoder()
    ztozd = ZtoZd()
    ztozg = ZtoZg()

    # encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation')
    # ztozd.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
    # ztozg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
    encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage1')
    ztozd.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
    ztozg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage1')

    path_AR = '/home/bosen/PycharmProjects/Datasets/AR_train/'


    zd_feature, zg_feature = [], []
    label = []
    images = []
    ID = [f"ID{i}" for i in range(1, 91)]
    for id in ID:
        for num, filename in enumerate(os.listdir(path_AR + id)):
            if num < 20:
                image = cv2.imread(path_AR + id + '/' + filename, 0) / 255
                low1, low2, low3 = downscale(image)
                images.append(low1), images.append(low2), images.append(low3)
                zdh, zdm, zdl, zgh, zgm, zgl = get_feature(image)
                zd_feature.append(zdh), zd_feature.append(zdm), zd_feature.append(zdl)
                zg_feature.append(zgh), zg_feature.append(zgm), zg_feature.append(zgl)
                label.append(int(id[2:])), label.append(int(id[2:])), label.append(int(id[2:]))

    test1_image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_train/ID1/1_train.jpg', 0) / 255
    test2_image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_train/ID90/1_train.jpg', 0) / 255
    zdh_test1, zdm_test1, zdl_test1, zgh_test1, zgm_test1, zgl_test1 = get_feature(test1_image)
    zdh_test2, zdm_test2, zdl_test2, zgh_test2, zgm_test2, zgl_test2 = get_feature(test2_image)
    zd_feature.append(zdh_test1), zd_feature.append(zdm_test1), zd_feature.append(zdl_test1), zd_feature.append(zdh_test2), zd_feature.append(zdm_test2), zd_feature.append(zdl_test2)
    zg_feature.append(zgh_test1), zg_feature.append(zgm_test1), zg_feature.append(zgl_test1), zg_feature.append(zgh_test2), zg_feature.append(zgm_test2), zg_feature.append(zgl_test2)
    zd_feature, zg_feature = np.array(zd_feature), np.array(zg_feature)

    print(zdh_test1.shape, zd_feature[60:5400].shape)
    zdh_test1 = tf.tile(tf.reshape(zdh_test1, [1, 200]), [zd_feature[60:5400].shape[0], 1])
    # similarity_zd = list(tf.reduce_mean(tf.square(zd_feature[60:5400] - zdh_test1), axis=-1))
    similarity_zd = list(cosine_similarity(zd_feature[60:5400], zdh_test1)[:, 0])
    zd_neighbor_index = similarity_zd.index(max(similarity_zd))+60
    print(zd_neighbor_index)

    print(zgh_test1, zgh_test2)
    zgh_test1 = tf.tile(tf.reshape(zgh_test1, [1, 200]), [zg_feature[60:5400].shape[0], 1])
    # similarity_zg = list(tf.reduce_mean(tf.square(zg_feature[60:5400] - zgh_test1), axis=-1))
    similarity_zg = list(cosine_similarity(zg_feature[60:5400], zgh_test1)[:, 0])
    zg_neighbor_index = similarity_zg.index(max(similarity_zg))+60
    print(zg_neighbor_index)




    pca = PCA(n_components=2)
    zd_feature = pca.fit_transform(zd_feature)

    # plt.figure(figsize=(7, 5))
    # plt.style.use('ggplot')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    # plt.title('Zd distribution')
    # plt.scatter(zd_feature[0: 1800][:, 0], zd_feature[0: 1800][:, 1], c='b', s=20)
    # plt.scatter(zd_feature[1800: 3600][:, 0], zd_feature[1800: 3600][:, 1], c='b', s=20)
    # plt.scatter(zd_feature[3600: 5400][:, 0], zd_feature[3600: 5400][:, 1], c='b', s=20)
    # plt.scatter(zd_feature[5400][0], zd_feature[5400][1], c='g', s=300, marker='*', alpha=1)
    # plt.scatter(zd_feature[5401][0], zd_feature[5401][1], c='k', s=300, marker='*', alpha=0.7)
    # plt.scatter(zd_feature[5402][0], zd_feature[5402][1], c='y', s=300, marker='*', alpha=0.5)
    # plt.scatter(zd_feature[5403][0], zd_feature[5403][1], c='g', s=300, marker='o', alpha=1)
    # plt.scatter(zd_feature[5404][0], zd_feature[5404][1], c='k', s=300, marker='o', alpha=0.7)
    # plt.scatter(zd_feature[5405][0], zd_feature[5405][1], c='y', s=300, marker='o', alpha=0.5)
    # plt.scatter(zd_feature[2571][0], zd_feature[2571][1], c='g', s=300, marker=6, alpha=1)
    # plt.scatter(zd_feature[2572][0], zd_feature[2572][1], c='k', s=300, marker=6, alpha=0.7)
    # plt.scatter(zd_feature[2573][0], zd_feature[2573][1], c='y', s=300, marker=6, alpha=0.5)
    # plt.show()
    # plt.close()


    pca = PCA(n_components=2)
    zg_feature = pca.fit_transform(zg_feature)

    plt.figure(figsize=(7, 5))
    plt.style.use('ggplot')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Zg distribution')
    plt.scatter(zg_feature[0: 1800][:, 0], zg_feature[0: 1800][:, 1], c='r', s=20)
    plt.scatter(zg_feature[1800: 3600][:, 0], zg_feature[1800: 3600][:, 1], c='r', s=20)
    plt.scatter(zg_feature[3600: 5400][:, 0], zg_feature[3600: 5400][:, 1], c='r', s=20)
    plt.scatter(zg_feature[5400][0], zg_feature[5400][1], c='g', s=300, marker="*", alpha=1)
    plt.scatter(zg_feature[5401][0], zg_feature[5401][1], c='k', s=300, marker="*", alpha=0.7)
    plt.scatter(zg_feature[5402][0], zg_feature[5402][1], c='y', s=300, marker="*", alpha=0.5)
    plt.scatter(zg_feature[5403][0], zg_feature[5403][1], c='g', s=300, marker="o", alpha=1)
    plt.scatter(zg_feature[5404][0], zg_feature[5404][1], c='k', s=300, marker="o", alpha=0.7)
    plt.scatter(zg_feature[5405][0], zg_feature[5405][1], c='y', s=300, marker="o", alpha=0.5)
    plt.scatter(zg_feature[1245][0], zg_feature[1245][1], c='g', s=300, marker=6, alpha=1)
    plt.scatter(zg_feature[1246][0], zg_feature[1246][1], c='k', s=300, marker=6, alpha=0.7)
    plt.scatter(zg_feature[1247][0], zg_feature[1247][1], c='y', s=300, marker=6, alpha=0.5)
    plt.show()
    plt.close()

def zg_distil_x_no_distill_space_find_hard_sample():
    global encoder
    global ztozd
    global ztozg

    encoder = encoder()
    ztozd = ZtoZd()
    ztozg = ZtoZg()
    def get_feature(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        z1, _ = encoder(low1_image)
        z2, _ = encoder(low2_image)
        z3, _ = encoder(low3_image)
        zgh, _, _ = ztozg(z1)
        zgm, _, _ = ztozg(z2)
        zgl, _, _ = ztozg(z3)
        return tf.reshape(zgh, [200]), tf.reshape(zgm, [200]), tf.reshape(zgl, [200])

    def downscale(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        return low1_image, low2_image, low3_image

    def get_whether_distillation_feautre(distillation=True):
        if distillation:
            encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation')
            ztozg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
        else:
            encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage1')
            ztozg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage1')

        path_AR = '/home/bosen/PycharmProjects/Datasets/AR_train/'

        zg_feature = []
        images = []
        ID = [f"ID{i}" for i in range(1, 91)]
        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR + id)):
                if num < 20:
                    image = cv2.imread(path_AR + id + '/' + filename, 0) / 255
                    low1, low2, low3 = downscale(image)
                    images.append(low1), images.append(low2), images.append(low3)
                    zgh, zgm, zgl = get_feature(image)
                    zg_feature.append(zgh), zg_feature.append(zgm), zg_feature.append(zgl)

        test1_image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_train/ID1/1_train.jpg', 0) / 255
        test2_image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_train/ID90/1_train.jpg', 0) / 255
        zgh_test1, zgm_test1, zgl_test1 = get_feature(test1_image)
        zgh_test2, zgm_test2, zgl_test2 = get_feature(test2_image)
        zg_feature.append(zgh_test1), zg_feature.append(zgm_test1), zg_feature.append(zgl_test1), zg_feature.append(zgh_test2), zg_feature.append(zgm_test2), zg_feature.append(zgl_test2)
        zg_feature = np.array(zg_feature)


        print(zgh_test1, zgh_test2)
        zgh_test1 = tf.tile(tf.reshape(zgh_test1, [1, 200]), [zg_feature[60:5400].shape[0], 1])
        # similarity_zg = list(tf.reduce_mean(tf.square(zg_feature[60:5400] - zgh_test1), axis=-1))
        similarity_zg = list(cosine_similarity(zg_feature[60:5400], zgh_test1)[:, 0])
        zg_neighbor_index = similarity_zg.index(max(similarity_zg)) + 60
        print(zg_neighbor_index)

        if distillation:
            return zg_feature

        else:
            return zg_feature, zg_neighbor_index


    zg_dis_feature = get_whether_distillation_feautre(distillation=True)
    zg_no_dis_feature, zg_no_dis_neighbor_index = get_whether_distillation_feautre(distillation=False)
    print(zg_no_dis_feature.shape, zg_dis_feature.shape, zg_no_dis_neighbor_index)

    pca = PCA(n_components=2)
    zg_dis_feature = pca.fit_transform(zg_dis_feature)
    zg_no_dis_feature = pca.transform(zg_no_dis_feature)



    plt.figure(figsize=(7, 5))
    plt.style.use('ggplot')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Zg distribution')
    plt.scatter(zg_dis_feature[0: 1800][:, 0], zg_dis_feature[0: 1800][:, 1], c='r', s=20)
    plt.scatter(zg_dis_feature[1800: 3600][:, 0], zg_dis_feature[1800: 3600][:, 1], c='r', s=20)
    plt.scatter(zg_dis_feature[3600: 5400][:, 0], zg_dis_feature[3600: 5400][:, 1], c='r', s=20)
    plt.scatter(zg_dis_feature[5400][0], zg_dis_feature[5400][1], c='g', s=300, marker="*", alpha=1)
    plt.scatter(zg_dis_feature[5401][0], zg_dis_feature[5401][1], c='k', s=300, marker="*", alpha=0.7)
    plt.scatter(zg_dis_feature[5402][0], zg_dis_feature[5402][1], c='y', s=300, marker="*", alpha=0.5)
    plt.scatter(zg_dis_feature[5403][0], zg_dis_feature[5403][1], c='g', s=300, marker="o", alpha=1)
    plt.scatter(zg_dis_feature[5404][0], zg_dis_feature[5404][1], c='k', s=300, marker="o", alpha=0.7)
    plt.scatter(zg_dis_feature[5405][0], zg_dis_feature[5405][1], c='y', s=300, marker="o", alpha=0.5)
    plt.scatter(zg_dis_feature[1245][0], zg_dis_feature[1245][1], c='g', s=300, marker=6, alpha=1)
    plt.scatter(zg_dis_feature[1246][0], zg_dis_feature[1246][1], c='k', s=300, marker=6, alpha=0.7)
    plt.scatter(zg_dis_feature[1247][0], zg_dis_feature[1247][1], c='y', s=300, marker=6, alpha=0.5)

    plt.scatter(zg_no_dis_feature[5400][0], zg_no_dis_feature[5400][1], c='b', s=300, marker="*", alpha=1)
    plt.scatter(zg_no_dis_feature[5401][0], zg_no_dis_feature[5401][1], c='b', s=300, marker="*", alpha=0.7)
    plt.scatter(zg_no_dis_feature[5402][0], zg_no_dis_feature[5402][1], c='b', s=300, marker="*", alpha=0.5)
    plt.scatter(zg_no_dis_feature[5403][0], zg_no_dis_feature[5403][1], c='b', s=300, marker="o", alpha=1)
    plt.scatter(zg_no_dis_feature[5404][0], zg_no_dis_feature[5404][1], c='b', s=300, marker="o", alpha=0.7)
    plt.scatter(zg_no_dis_feature[5405][0], zg_no_dis_feature[5405][1], c='b', s=300, marker="o", alpha=0.5)
    plt.scatter(zg_no_dis_feature[1245][0], zg_no_dis_feature[1245][1], c='b', s=300, marker=6, alpha=1)
    plt.scatter(zg_no_dis_feature[1246][0], zg_no_dis_feature[1246][1], c='b', s=300, marker=6, alpha=0.7)
    plt.scatter(zg_no_dis_feature[1247][0], zg_no_dis_feature[1247][1], c='b', s=300, marker=6, alpha=0.5)
    plt.show()
    plt.close()

def compare_forward_inversion_image():
    def downscale(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low2_image = cv2.resize(cv2.resize(image, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low3_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low4_image = cv2.resize(cv2.resize(image, (12, 12), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        low5_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        return low1_image, low2_image, low3_image, low4_image, low5_image

    def gen_image(image):
        low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low2_image = cv2.resize(cv2.resize(image, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low3_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low4_image = cv2.resize(cv2.resize(image, (12, 12), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
        low5_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

        z32, _ = encoder(low1_image)
        z20, _ = encoder(low2_image)
        z16, _ = encoder(low3_image)
        z12, _ = encoder(low4_image)
        z8, _ = encoder(low5_image)

        zg32, _, _ = ztozg(z32)
        zg20, _, _ = ztozg(z20)
        zg16, _, _ = ztozg(z16)
        zgl2, _, _ = ztozg(z12)
        zg8, _, _ = ztozg(z8)

        gen1_image = tf.reshape(generator(zg32), [64, 64])
        gen2_image = tf.reshape(generator(zg20), [64, 64])
        gen3_image = tf.reshape(generator(zg16), [64, 64])
        gen4_image = tf.reshape(generator(zgl2), [64, 64])
        gen5_image = tf.reshape(generator(zg8), [64, 64])

        return gen1_image, gen2_image, gen3_image, gen4_image, gen5_image

    global encoder
    global ztozd
    global ztozg
    global generator

    encoder = encoder()
    ztozd = ZtoZd()
    ztozg = ZtoZg()
    generator = generator()

    encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation')
    ztozg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
    generator.load_weights("/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation")

    path = "/home/bosen/gradation_thesis/synthesis_system/cls_datasets/train_data_var_large/"

    ID = ['ID1', 'ID2', 'ID3', 'ID88', 'ID89', 'ID90']
    plt.subplots(figsize=(5, 25))
    plt.subplots_adjust(hspace=0, wspace=0)
    for num, id in enumerate(ID):
        for filename in os.listdir(path + id):
            image = cv2.imread(path + id + '/' + filename, 0) / 255
            if 'bmp' in filename:
                plt.subplot(16, 6, num + 1)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
                low_image1, low_image2, low_image3, low_image4, low_image5 = downscale(image)
                low = [low_image1, low_image2, low_image3, low_image4, low_image5]
                for i in range(5):
                    plt.subplot(16, 6, num + 1 + ((i+1) * 6))
                    plt.axis('off')
                    plt.imshow(low[i], cmap='gray')
                gen_image1, gen_image2, gen_image3, gen_image4, gen_image5 = gen_image(image)
                gen = [gen_image1, gen_image2, gen_image3, gen_image4, gen_image5]
                for i in range(5):
                    plt.subplot(16, 6, num + 1 + ((i+6) * 6))
                    plt.axis('off')
                    plt.imshow(gen[i], cmap='gray')
            elif '32' in filename:
                plt.subplot(16, 6, num + 67)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
            elif '20' in filename:
                plt.subplot(16, 6, num + 73)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
            elif '16' in filename:
                plt.subplot(16, 6, num + 79)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
            elif '12' in filename:
                plt.subplot(16, 6, num + 85)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
            elif '8' in filename:
                plt.subplot(16, 6, num + 91)
                plt.axis('off')
                plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()

def distillation_correlation_matrix():
    global with_distillation_encoder
    global without_distillation_encoder
    global ztozd
    global with_distillation_ztozg
    global without_distillation_ztozg

    with_distillation_encoder = normal_encoder()
    without_distillation_encoder = normal_encoder()
    ztozd = ZtoZd()
    with_distillation_ztozg = ZtoZg()
    without_distillation_ztozg = ZtoZg()

    with_distillation_encoder.load_weights('model_weight/AE_encoder')
    without_distillation_encoder.load_weights('model_weight/AE_encoder')
    ztozd.load_weights('model_weight/AE_ztozd')
    with_distillation_ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    without_distillation_ztozg.load_weights('model_weight/patch_ztozg')

    path_AR_syn = '/home/bosen/PycharmProjects/Datasets/AR_train/'
    ID = [f'ID{i}' for i in [9, 10, 11, 12, 82, 81, 80, 79]]

    zds, zghs_with_distillation, zgms_with_distillation, zgls_with_distillationzghs, zghs_without_distillation, zgms_without_distillation, zgls_without_distillation = [], [], [], [], [], [], []
    for id in ID:
        for num, filename in enumerate(os.listdir(path_AR_syn + id)):
            if num < 5:
                image = cv2.resize(cv2.imread(path_AR_syn + id + '/' + filename, 0)/255, (64, 64))
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                zH, zh_with_distillation, zm_with_distillation, zl_with_distillation = with_distillation_encoder(image.reshape(1, 64, 64, 1)), with_distillation_encoder(low1_image.reshape(1, 64, 64, 1)), with_distillation_encoder(low2_image.reshape(1, 64, 64, 1)), with_distillation_encoder(low3_image.reshape(1, 64, 64, 1))
                zh_without_distillation, zm_without_distillation, zl_without_distillation = without_distillation_encoder(low1_image.reshape(1, 64, 64, 1)), without_distillation_encoder(low2_image.reshape(1, 64, 64, 1)), without_distillation_encoder(low3_image.reshape(1, 64, 64, 1))

                zd, _ = ztozd(zH)
                zgh_with_distillation, _, _ = with_distillation_ztozg(zh_with_distillation)
                zgm_with_distillation, _, _ = with_distillation_ztozg(zm_with_distillation)
                zgl_with_distillation, _, _ = with_distillation_ztozg(zl_with_distillation)
                zgh_without_distillation, _, _ = without_distillation_ztozg(zh_without_distillation)
                zgm_without_distillation, _, _ = without_distillation_ztozg(zm_without_distillation)
                zgl_without_distillation, _, _ = without_distillation_ztozg(zl_without_distillation)

                zds.append(tf.reshape(zd, [200]))
                zghs_with_distillation.append(tf.reshape(zgh_with_distillation, [200]))
                zgms_with_distillation.append(tf.reshape(zgm_with_distillation, [200]))
                zgls_with_distillationzghs.append(tf.reshape(zgl_with_distillation, [200]))
                zghs_without_distillation.append(tf.reshape(zgh_without_distillation, [200]))
                zgms_without_distillation.append(tf.reshape(zgm_without_distillation, [200]))
                zgls_without_distillation.append(tf.reshape(zgl_without_distillation, [200]))
            else:
                break

    name = ['zdH-distribution', 'w-distillation-zgh-distribution', 'w-distillation-zgm-distribution', 'w-distillation-zgl-distribution', 'wo-distillation-zgh-distribution', 'wo-distllation-zgm-distribution', 'wo-distillation-zgl-distribution']
    features = [zds, zghs_with_distillation, zgms_with_distillation, zgls_with_distillationzghs, zghs_without_distillation, zgms_without_distillation, zgls_without_distillation]

    for num, feature in enumerate(features):
        aff_mtx_zd = tf.matmul(feature, tf.transpose(feature)) / (tf.matmul(tf.norm(feature, axis=1, keepdims=True), tf.norm(tf.transpose(feature), axis=0, keepdims=True)))
        # H_trans_zd = aff_mtx_zd / tf.reshape(tf.reduce_sum(aff_mtx_zd, axis=1), [aff_mtx_zd.get_shape()[0], 1])
        plt.title(f'{name[num]}')
        sns.heatmap(aff_mtx_zd, vmin=-1, vmax=1)
        plt.savefig(f'result/synthesis_experiment/distillation_experiment/{name[num]}')
        plt.close()

def compare_with_distllation_or_without_distillation_forward_answer():
    global with_distillation_encoder
    global with_distillation_ztozg
    global with_distillation_generator

    global without_distillation_encoder
    global without_distillation_ztozg
    global without_distillation_generator

    with_distillation_encoder = normal_encoder()
    with_distillation_ztozg = ZtoZg()
    with_distillation_generator = generator()

    without_distillation_encoder = normal_encoder()
    without_distillation_ztozg = ZtoZg()
    without_distillation_generator = generator()

    with_distillation_encoder.load_weights('model_weight/AE_encoder')
    with_distillation_ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    with_distillation_generator.load_weights('model_weight/zd_zg_distillation_generator')

    without_distillation_encoder.load_weights('model_weight/AE_encoder')
    without_distillation_ztozg.load_weights('model_weight/patch_ztozg')
    without_distillation_generator.load_weights('model_weight/patch_g')

    with_distillation_psnr1, with_distillation_psnr2, with_distillation_psnr3 = [], [], []
    without_distillation_psnr1, without_distillation_psnr2, without_distillation_psnr3 = [], [], []
    with_distillation_ssim1, with_distillation_ssim2, with_distillation_ssim3 = [], [], []
    without_distillation_ssim1, without_distillation_ssim2, without_distillation_ssim3 = [], [], []

    path_AR_real_test = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_train90/"
    # path_AR_real_test = "/disk2/DCGAN_yu/CK1/ARtest/"
    path_AR_real_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

    plt.subplots(figsize=(7, 10))
    plt.subplots_adjust(hspace=0, wspace=0)
    count = 0
    for num, ID in enumerate(os.listdir(path_AR_real_test)):
        for filename in os.listdir(path_AR_real_test + ID):
    # for num, filename in enumerate(os.listdir(path_AR_real_test)):
            if '13_test' in filename:
                # image = cv2.imread(path_AR_real_test + ID + '/' + filename, 0) / 255
                image = cv2.imread(path_AR_real_test + ID + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                low1_image = cv2.resize(image, (32, 32), cv2.INTER_CUBIC)
                low2_image = cv2.resize(image, (16, 16), cv2.INTER_CUBIC)
                low3_image = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
                low1_image = cv2.resize(low1_image, (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(low2_image, (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(low3_image, (64, 64), cv2.INTER_CUBIC)

                z1 = with_distillation_encoder(low1_image.reshape(1, 64, 64, 1))
                z2 = with_distillation_encoder(low2_image.reshape(1, 64, 64, 1))
                z3 = with_distillation_encoder(low3_image.reshape(1, 64, 64, 1))
                zg1, _, _ = with_distillation_ztozg(z1)
                zg2, _, _ = with_distillation_ztozg(z2)
                zg3, _, _ = with_distillation_ztozg(z3)
                with_distillation_gen1_image = with_distillation_generator(zg1)
                with_distillation_gen2_image = with_distillation_generator(zg2)
                with_distillation_gen3_image = with_distillation_generator(zg3)

                z1 = without_distillation_encoder(low1_image.reshape(1, 64, 64, 1))
                z2 = without_distillation_encoder(low2_image.reshape(1, 64, 64, 1))
                z3 = without_distillation_encoder(low3_image.reshape(1, 64, 64, 1))
                zg1, _, _ = without_distillation_ztozg(z1)
                zg2, _, _ = without_distillation_ztozg(z2)
                zg3, _, _ = without_distillation_ztozg(z3)
                without_distillation_gen1_image = without_distillation_generator(zg1)
                without_distillation_gen2_image = without_distillation_generator(zg2)
                without_distillation_gen3_image = without_distillation_generator(zg3)

                with_distillation_psnr1.append(tf.image.psnr(image.reshape(1, 64, 64, 1), with_distillation_gen1_image, max_val=1))
                with_distillation_psnr2.append(tf.image.psnr(image.reshape(1, 64, 64, 1), with_distillation_gen2_image, max_val=1))
                with_distillation_psnr3.append(tf.image.psnr(image.reshape(1, 64, 64, 1), with_distillation_gen3_image, max_val=1))
                without_distillation_psnr1.append(tf.image.psnr(image.reshape(1, 64, 64, 1), without_distillation_gen1_image, max_val=1))
                without_distillation_psnr2.append(tf.image.psnr(image.reshape(1, 64, 64, 1), without_distillation_gen2_image, max_val=1))
                without_distillation_psnr3.append(tf.image.psnr(image.reshape(1, 64, 64, 1), without_distillation_gen3_image, max_val=1))


                with_distillation_ssim1.append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(with_distillation_gen1_image, dtype=tf.float32), max_val=1))
                with_distillation_ssim2.append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(with_distillation_gen2_image, dtype=tf.float32), max_val=1))
                with_distillation_ssim3.append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(with_distillation_gen3_image, dtype=tf.float32), max_val=1))
                without_distillation_ssim1.append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(without_distillation_gen1_image, dtype=tf.float32), max_val=1))
                without_distillation_ssim2.append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(without_distillation_gen2_image, dtype=tf.float32), max_val=1))
                without_distillation_ssim3.append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(without_distillation_gen3_image, dtype=tf.float32), max_val=1))

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
                plt.imshow(tf.reshape(with_distillation_gen1_image, [64, 64]), cmap='gray')

                plt.subplot(10, 7, count + 36)
                plt.axis('off')
                plt.imshow(tf.reshape(with_distillation_gen2_image, [64, 64]), cmap='gray')

                plt.subplot(10, 7, count + 43)
                plt.axis('off')
                plt.imshow(tf.reshape(with_distillation_gen3_image, [64, 64]), cmap='gray')

                plt.subplot(10, 7, count + 50)
                plt.axis('off')
                plt.imshow(tf.reshape(without_distillation_gen1_image, [64, 64]), cmap='gray')

                plt.subplot(10, 7, count + 57)
                plt.axis('off')
                plt.imshow(tf.reshape(without_distillation_gen2_image, [64, 64]), cmap='gray')

                plt.subplot(10, 7, count + 64)
                plt.axis('off')
                plt.imshow(tf.reshape(without_distillation_gen3_image, [64, 64]), cmap='gray')
                count += 1


                if (num + 1) % 7 == 0:
                    plt.savefig(f'result/synthesis_experiment/distillation_experiment/{num}_image')
                    plt.close()
                    plt.subplots(figsize=(7, 10))
                    plt.subplots_adjust(hspace=0, wspace=0)
                    count = 0

    x1 = [0.8, 1.8, 2.8]
    x2 = [1, 2, 3]

    with_distillation_psnr = [np.mean(with_distillation_psnr1), np.mean(with_distillation_psnr2), np.mean(with_distillation_psnr3)]
    without_distillation_psnr = [np.mean(without_distillation_psnr1), np.mean(without_distillation_psnr2), np.mean(without_distillation_psnr3)]
    with_distillaiton_ssim = [np.mean(with_distillation_ssim1), np.mean(with_distillation_ssim2), np.mean(with_distillation_ssim3)]
    without_distillation_ssim = [np.mean(without_distillation_ssim1), np.mean(without_distillation_ssim2), np.mean(without_distillation_ssim3)]

    plt.bar(x1, with_distillation_psnr, color='b', width=0.2)
    plt.bar(x2, without_distillation_psnr, color='r', width=0.2, align='edge')
    plt.legend(['With Distillation MPSNR', 'Without Distillation mPSNR'], loc='lower left')
    plt.title('PSNR')
    bars = ("", '2-ratio', '4-ratio', '8-ratio')
    x_pos = np.arange(len(bars))
    plt.xticks(x_pos, bars)
    plt.savefig(f'result/synthesis_experiment/distillation_experiment/with_or_witout_distillation_MPSNR')
    plt.close()

    plt.bar(x1, with_distillaiton_ssim, color='b', width=0.2)
    plt.bar(x2, without_distillation_ssim, color='r', width=0.2, align='edge')
    plt.legend(['With Distillation MSSIM', 'Without Distillation mSSIM'], loc='lower left')
    plt.title('SSIM')
    bars = ("", '2-ratio', '4-ratio', '8-ratio')
    x_pos = np.arange(len(bars))
    plt.xticks(x_pos, bars)
    plt.savefig(f'result/synthesis_experiment/distillation_experiment/with_or_witout_distillation_MSSIM')
    plt.close()

    print(with_distillation_psnr1, with_distillation_psnr2, with_distillation_psnr3)
    print(without_distillation_psnr1, without_distillation_psnr2, without_distillation_psnr3)

    print(with_distillation_ssim1, with_distillation_ssim2, with_distillation_ssim3)
    print(without_distillation_ssim1, without_distillation_ssim2, without_distillation_ssim3)

    print(np.mean(with_distillation_psnr1), np.mean(with_distillation_psnr2), np.mean(with_distillation_psnr3))
    print(np.mean(without_distillation_psnr1), np.mean(without_distillation_psnr2), np.mean(without_distillation_psnr3))

    print(np.mean(with_distillation_ssim1), np.mean(with_distillation_ssim2), np.mean(with_distillation_ssim3))
    print(np.mean(without_distillation_ssim1), np.mean(without_distillation_ssim2), np.mean(without_distillation_ssim3))

def cls_with_reid_and_without_reid_heatmap(aug=True):
    def gradcam_heatmap_mutiple(img_array, model, last_conv_layer_name, network_layer_name, label, corresponding_label=True):
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
                pred_index = tf.constant([label], dtype=tf.int64)
            else:
                pred_index = np.argmax(preds, axis=-1)
            class_channel = tf.gather(preds, pred_index, axis=-1, batch_dims=1)
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        pooled_gradsa = tf.tile(tf.reshape(pooled_grads, [pooled_grads.shape[0], 1, 1, pooled_grads.shape[1]]),
                                [1, last_conv_layer_output.shape[1], last_conv_layer_output.shape[2], 1])
        heatmap = last_conv_layer_output * pooled_gradsa
        heatmap = tf.reduce_sum(heatmap, axis=-1)
        heatmap_min, heatmap_max = [], []
        for num in range(heatmap.shape[0]):
            heatmap_min.append(np.min(heatmap[num]))
            heatmap_max.append(np.max(heatmap[num]))
        heatmap_min, heatmap_max = tf.cast(heatmap_min, dtype=tf.float32), tf.cast(heatmap_max, dtype=tf.float32)
        heatmap_min = tf.tile(tf.reshape(heatmap_min, [heatmap_min.shape[0], 1, 1]),
                              [1, heatmap.shape[1], heatmap.shape[2]])
        heatmap_max = tf.tile(tf.reshape(heatmap_max, [heatmap_max.shape[0], 1, 1]),
                              [1, heatmap.shape[1], heatmap.shape[2]])
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-20)

        heatmap_gray = tf.cast(heatmap, dtype=tf.float32)
        heatmap = np.uint8(255 * heatmap)

        cmap = cm.get_cmap("jet")
        cmap_colors = cmap(np.arange(256))[:, :3]
        cmap_heatmap = cmap_colors[heatmap]
        cmap_heatmap = tf.image.resize(cmap_heatmap, [64, 64], method='bicubic')
        heatmap_gray = tf.image.resize(tf.reshape(heatmap_gray, [-1, 16, 16, 1]), [64, 64], method='bicubic')
        return cmap_heatmap, heatmap_gray

    global encoder
    global ztozg
    global generator
    global cls_reid
    global cls

    encoder = normal_encoder()
    ztozg = ZtoZg()
    generator = generator()
    cls_reid = classifier()
    cls = classifier()

    encoder.load_weights('model_weight/AE_encoder')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    generator.load_weights('model_weight/zd_zg_distillation_generator')
    if aug:
        cls.load_weights('model_weight/cls')
        cls_reid.load_weights('model_weight/reid_cls_aug')
    else:
        cls.load_weights('model_weight/cls')
        cls_reid.load_weights('model_weight/reid_cls')

    for layer in cls.layers:
        print(layer.name)
    for layer in cls_reid.layers:
        print(layer.name)

    cls_last_conv_layer_name = 'conv2d_27'
    cls_network_layer_name = ['max_pooling2d_3', "conv2d_28", 'max_pooling2d_4', 'conv2d_29', 'max_pooling2d_5', 'flatten_2', 'dense_8', 'dropout_4', 'dense_9']

    cls_reid_last_conv_layer_name = 'conv2d_24'
    cls_reid_network_layer_name = ['max_pooling2d', "conv2d_25", 'max_pooling2d_1', 'conv2d_26', 'max_pooling2d_2', 'flatten_1', 'dense_6', 'dropout_3', 'dense_7']

    path = '/home/bosen/PycharmProjects/Datasets/AR_test/'
    num = 0
    plt.subplots(figsize=(7, 13))
    plt.subplots_adjust(hspace=0, wspace=0)
    for num_id, id in enumerate(os.listdir(path)):
        for count, filename in enumerate(os.listdir(path + id)):
            if count == 21:
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                low1_test = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_test = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_test = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                z1, z2, z3 = encoder(low1_test), encoder(low2_test), encoder(low3_test)
                zg1, _, _ = ztozg(z1)
                zg2, _, _ = ztozg(z2)
                zg3, _, _ = ztozg(z3)

                forward_low1 = generator(zg1)
                forward_low2 = generator(zg2)
                forward_low3 = generator(zg3)

                attention_low1, _ = gradcam_heatmap_mutiple(forward_low1, cls, cls_last_conv_layer_name, cls_network_layer_name, label=tf.one_hot(int(id[2:])-1, 111), corresponding_label=True)
                attention_low2, _ = gradcam_heatmap_mutiple(forward_low2, cls, cls_last_conv_layer_name, cls_network_layer_name, label=tf.one_hot(int(id[2:])-1, 111), corresponding_label=True)
                attention_low3, _ = gradcam_heatmap_mutiple(forward_low3, cls, cls_last_conv_layer_name, cls_network_layer_name, label=tf.one_hot(int(id[2:])-1, 111), corresponding_label=True)

                attention_low1_reid, _ = gradcam_heatmap_mutiple(forward_low1, cls_reid, cls_reid_last_conv_layer_name, cls_reid_network_layer_name, label=tf.one_hot(int(id[2:]) - 1, 111), corresponding_label=True)
                attention_low2_reid, _ = gradcam_heatmap_mutiple(forward_low2, cls_reid, cls_reid_last_conv_layer_name, cls_reid_network_layer_name, label=tf.one_hot(int(id[2:]) - 1, 111), corresponding_label=True)
                attention_low3_reid, _ = gradcam_heatmap_mutiple(forward_low3, cls_reid, cls_reid_last_conv_layer_name, cls_reid_network_layer_name, label=tf.one_hot(int(id[2:]) - 1, 111), corresponding_label=True)

                plt.subplot(13, 7, num + 1)
                plt.axis('off')
                plt.imshow(image, cmap='gray')

                plt.subplot(13, 7, num + 8)
                plt.axis('off')
                plt.imshow(tf.reshape(low1_test, [64, 64]), cmap='gray')
                plt.subplot(13, 7, num + 15)
                plt.axis('off')
                plt.imshow(tf.reshape(low2_test, [64, 64]), cmap='gray')
                plt.subplot(13, 7, num + 22)
                plt.axis('off')
                plt.imshow(tf.reshape(low3_test, [64, 64]), cmap='gray')

                plt.subplot(13, 7, num + 29)
                plt.axis('off')
                plt.imshow(tf.reshape(forward_low1, [64, 64]), cmap='gray')
                plt.subplot(13, 7, num + 36)
                plt.axis('off')
                plt.imshow(tf.reshape(forward_low2, [64, 64]), cmap='gray')
                plt.subplot(13, 7, num + 43)
                plt.axis('off')
                plt.imshow(tf.reshape(forward_low3, [64, 64]), cmap='gray')

                plt.subplot(13, 7, num + 50)
                plt.axis('off')
                plt.imshow(tf.reshape(attention_low1, [64, 64, 3]), cmap='gray')
                plt.subplot(13, 7, num + 57)
                plt.axis('off')
                plt.imshow(tf.reshape(attention_low2, [64, 64, 3]), cmap='gray')
                plt.subplot(13, 7, num + 64)
                plt.axis('off')
                plt.imshow(tf.reshape(attention_low3, [64, 64, 3]), cmap='gray')

                plt.subplot(13, 7, num + 71)
                plt.axis('off')
                plt.imshow(tf.reshape(attention_low1_reid, [64, 64, 3]), cmap='gray')
                plt.subplot(13, 7, num + 78)
                plt.axis('off')
                plt.imshow(tf.reshape(attention_low2_reid, [64, 64, 3]), cmap='gray')
                plt.subplot(13, 7, num + 85)
                plt.axis('off')
                plt.imshow(tf.reshape(attention_low3_reid, [64, 64, 3]), cmap='gray')
                num += 1
                if num % 7 == 0:
                    plt.savefig(f'result/cls/compare_with_reid_without_reid_with_augmentation_{aug}_{num_id}')
                    plt.close()
                    num = 0
                    plt.subplots(figsize=(7, 13))
                    plt.subplots_adjust(hspace=0, wspace=0)

def inversion_vs_forward_psnr_ssim(id):
    global encoder
    global ztozg
    global generator

    encoder = normal_encoder()
    ztozg = ZtoZg()
    generator = generator()

    encoder.load_weights('model_weight/AE_encoder')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    generator.load_weights('model_weight/zd_zg_distillation_generator')

    inversion_path1 = f'cls_data/test_data_inversion/ID{id}/11_test.jpg_32_resolution.jpg'
    inversion_path2 = f'cls_data/test_data_inversion/ID{id}/11_test.jpg_16_resolution.jpg'
    inversion_path3 = f'cls_data/test_data_inversion/ID{id}/11_test.jpg_8_resolution.jpg'
    if id-90 < 10:
        gt_path = f'/home/bosen/PycharmProjects/Datasets/AR_test/ID0{id-90}/11_test.jpg'
    else:
        gt_path = f'/home/bosen/PycharmProjects/Datasets/AR_test/ID{id - 90}/11_test.jpg'

    gt = cv2.imread(gt_path, 0) / 255
    gt = cv2.resize(gt, (64, 64), cv2.INTER_CUBIC)

    low1_gt = cv2.resize(gt, (32, 32), cv2.INTER_CUBIC)
    low2_gt = cv2.resize(gt, (16, 16), cv2.INTER_CUBIC)
    low3_gt = cv2.resize(gt, (8, 8), cv2.INTER_CUBIC)

    inversion_image1 = cv2.imread(inversion_path1, 0) / 255
    inversion_image2 = cv2.imread(inversion_path2, 0) / 255
    inversion_image3 = cv2.imread(inversion_path3, 0) / 255

    low1 = cv2.resize(cv2.resize(gt, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    low2 = cv2.resize(cv2.resize(gt, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    low3 = cv2.resize(cv2.resize(gt, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

    z1, z2, z3 = encoder(low1.reshape(1, 64, 64, 1)), encoder(low2.reshape(1, 64, 64, 1)), encoder(low3.reshape(1, 64, 64, 1))
    zg1, _, _ = ztozg(z1)
    zg2, _, _ = ztozg(z2)
    zg3, _, _ = ztozg(z3)

    syn1, syn2, syn3 = generator(zg1), generator(zg2), generator(zg3)
    inversion1, inversion2, inversion3 = inversion_image1.reshape(1, 64, 64, 1), inversion_image2.reshape(1, 64, 64, 1), inversion_image3.reshape(1, 64, 64, 1)

    low_psnr = [tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(low1.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0],
                tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(low2.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0],
                tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(low3.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0]]

    low_ssim = [tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(low1.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0],
                tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(low2.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0],
                tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(low3.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0]]

    forward_psnr = [tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), syn1, max_val=1)[0],
                    tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), syn2, max_val=1)[0],
                    tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), syn3, max_val=1)[0]]

    forward_ssim = [tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), syn1, max_val=1)[0],
                    tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), syn2, max_val=1)[0],
                    tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), syn3, max_val=1)[0]]

    inv_psnr = [tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), inversion1, max_val=1)[0],
                    tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), inversion2, max_val=1)[0],
                    tf.image.psnr(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), inversion3, max_val=1)[0]]

    inv_ssim = [tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(inversion1, dtype=tf.float32), max_val=1)[0],
                    tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(inversion2, dtype=tf.float32), max_val=1)[0],
                    tf.image.ssim(tf.cast(gt.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(inversion3, dtype=tf.float32), max_val=1)[0]]

    low_psnr = [round(num, 1) for num in np.array(low_psnr)]
    forward_psnr = [round(num, 1) for num in np.array(forward_psnr)]
    inv_psnr = [round(num, 1) for num in np.array(inv_psnr)]

    low_ssim = [round(num, 2) for num in np.array(low_ssim)]
    forward_ssim = [round(num, 2) for num in np.array(forward_ssim)]
    inv_ssim = [round(num, 2) for num in np.array(inv_ssim)]

    x1 = np.array([1, 2, 3])
    bar_width = 0.35
    x2 = x1 + bar_width
    x3 = x2 + bar_width

    #PSNR
    plt.bar(x1, low_psnr, width=bar_width, label='low')
    plt.bar(x2, forward_psnr, width=bar_width, label='Forward')
    plt.bar(x3, inv_psnr, width=bar_width, label='Inversion')

    for i, (val1, val2, val3) in enumerate(zip(low_psnr, forward_psnr, inv_psnr)):
        plt.text(x1[i], val1, str(val1)[0:4], ha='center', va='bottom')
        plt.text(x2[i], val2, str(val2)[0:4], ha='center', va='bottom')
        plt.text(x3[i], val3, str(val3)[0:4], ha='center', va='bottom')


    plt.title('PSNR')
    # plt.xlabel('Ratio')
    plt.ylabel('Value')
    plt.legend()

    plt.xticks(x1 + bar_width / 2, x1)
    plt.savefig(f'result/synthesis_experiment/ID{id}_forward(w_distillation)_vs_inversion_synthesis_PSNR')
    plt.close()

    #SSIM
    plt.bar(x1, low_ssim, width=bar_width, label='low')
    plt.bar(x2, forward_ssim, width=bar_width, label='Forward')
    plt.bar(x3, inv_ssim, width=bar_width, label='Inversion')

    for i, (val1, val2, val3) in enumerate(zip(low_ssim, forward_ssim, inv_ssim)):
        plt.text(x1[i], val1, str(val1)[0:4], ha='center', va='bottom')
        plt.text(x2[i], val2, str(val2)[0:4], ha='center', va='bottom')
        plt.text(x3[i], val1, str(val3)[0:4], ha='center', va='bottom')


    plt.title('SSIM')
    # plt.xlabel('Ratio')
    plt.ylabel('Value')
    # plt.legend()

    plt.xticks(x1 + bar_width / 2, x1)
    plt.savefig(f'result/synthesis_experiment/ID{id}_forward(w_distillation)_vs_inversion_synthesis_SSIM')
    plt.close()

def inversion_vs_forward_mpsnr_mssim():
    global encoder
    global ztozg
    global generator

    encoder = normal_encoder()
    ztozg = ZtoZg()
    generator = generator()

    encoder.load_weights('model_weight/AE_encoder')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    generator.load_weights('model_weight/zd_zg_distillation_generator')


    path_forward = '/home/bosen/PycharmProjects/Datasets/AR_test/'
    path_inversion = 'cls_data/test_data_inversion/'

    forward = [[] for i in range(3)]
    inversion = [[] for i in range(3)]
    high_image = []
    for id in os.listdir(path_forward):
        for filename in os.listdir(path_forward + id):
            if "11_test" in filename:
                image = cv2.imread(path_forward + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                high_image.append(image.reshape(64, 64, 1))
                low1 = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2 = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3 = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                z1, z2, z3 = encoder(low1), encoder(low2), encoder(low3)
                zg1, _, _ = ztozg(z1)
                zg2, _, _ = ztozg(z2)
                zg3, _, _ = ztozg(z3)
                for_syn1, for_syn2, for_syn3 = generator(zg1), generator(zg2), generator(zg3)
                forward[0].append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(for_syn1, dtype=tf.float32), max_val=1)[0])
                forward[1].append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(for_syn2, dtype=tf.float32), max_val=1)[0])
                forward[2].append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(for_syn3, dtype=tf.float32), max_val=1)[0])

    for id in os.listdir(path_inversion):
        for filename in os.listdir(path_inversion + id ):
            if "11_test.jpg_8" in filename:
                if int(id[2:]) - 90 < 10:
                    for_id = f"ID0{int(id[2:]) - 90}"
                else:
                    for_id = f"ID{int(id[2:]) - 90}"
                image = cv2.imread(path_forward + for_id + '/' + "11_test.jpg", 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                inv_image = cv2.imread(path_inversion + id + '/' + filename, 0) / 255
                inversion[0].append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(inv_image.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0])
            if "11_test.jpg_16" in filename:
                if int(id[2:]) - 90 < 10:
                    for_id = f"ID0{int(id[2:]) - 90}"
                else:
                    for_id = f"ID{int(id[2:]) - 90}"
                image = cv2.imread(path_forward + for_id + '/' + "11_test.jpg", 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                inv_image = cv2.imread(path_inversion + id + '/' + filename, 0) / 255
                inversion[1].append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(inv_image.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0])
            if "11_test.jpg_32" in filename:
                if int(id[2:]) - 90 < 10:
                    for_id = f"ID0{int(id[2:]) - 90}"
                else:
                    for_id = f"ID{int(id[2:]) - 90}"
                image = cv2.imread(path_forward + for_id + '/' + "11_test.jpg", 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                inv_image = cv2.imread(path_inversion + id + '/' + filename, 0) / 255
                inversion[2].append(tf.image.ssim(tf.cast(image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(inv_image.reshape(1, 64, 64, 1), dtype=tf.float32), max_val=1)[0])
    # forward_psnr = [round(num, 1) for num in np.array(forward)]
    # inversion_psnr = [round(num, 1) for num in np.array(inversion)]
    print(np.mean(forward[0]), np.mean(forward[1]), np.mean(forward[2]))
    print(np.mean(inversion[0]), np.mean(inversion[1]), np.mean(inversion[2]))

    x1 = [0.8, 1.8, 2.8]
    x2 = [1, 2, 3]

    forward_psnr = [np.mean(forward[0]), np.mean(forward[1]), np.mean(forward[1])]
    inversion_psnr = [np.mean(inversion[0]), np.mean(inversion[1]), np.mean(inversion[2])]


    plt.bar(x1, forward_psnr, color='b', width=0.2)
    plt.bar(x2, inversion_psnr, color='r', width=0.2, align='edge')
    plt.legend(['forward mSSIM', 'Inversion mSSIM'], loc='lower left')
    plt.title('mSSIM')
    bars = ("", '8-ratio', '4-ratio', '2-ratio')
    x_pos = np.arange(len(bars))
    plt.xticks(x_pos, bars)
    plt.show()


def with_instance_normalization_code_range_vs_gt_code_range():
    global encoder
    global ztozg
    global regression
    global generator

    encoder = normal_encoder()
    ztozg = ZtoZg()
    regression = regression_model_with_instance()
    encoder.load_weights('model_weight/AE_encoder')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    regression.load_weights('model_weight/regression_fine_tune')

    zgHs, zregHs, zreghs, zregms, zregls = [], [], [], [], []
    path = '/home/bosen/PycharmProjects/Datasets/AR_train/'
    for id in os.listdir(path):
        for count, filename in enumerate(os.listdir(path + id)):
            if count < 10:
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                zH = encoder(image.reshape(1, 64, 64, 1))
                zh = encoder(low1_image.reshape(1, 64, 64, 1))
                zm = encoder(low2_image.reshape(1, 64, 64, 1))
                zl = encoder(low3_image.reshape(1, 64, 64, 1))
                zgH, _, _ = ztozg(zH)
                zgh, _, _ = ztozg(zh)
                zgm, _, _ = ztozg(zm)
                zgl, _, _ = ztozg(zl)

                zgHs.append(tf.reshape(zgH, [200]))

                zregH = regression(zgH)
                zregh = regression(zgh)
                zregm = regression(zgm)
                zregl = regression(zgl)

                zregHs.append(tf.reshape(zregH, [200]))
                zreghs.append(tf.reshape(zregh, [200]))
                zregms.append(tf.reshape(zregm, [200]))
                zregls.append(tf.reshape(zregl, [200]))

    zgHs, zregHs, zreghs, zregms, zregls = np.array(zgHs), np.array(zregHs), np.array(zreghs), np.array(zregms), np.array(zregls)
    zgHs, zregHs, zreghs, zregms, zregls = zgHs.reshape(-1), zregHs.reshape(-1), zreghs.reshape(-1), zregms.reshape(-1), zregls.reshape(-1)

    for (name, reg) in zip(['zregH', 'zregh', 'zregm', 'zregl'], [zregHs, zreghs, zregms, zregls]):
        plt.hist(zgHs)
        plt.hist(reg, alpha=0.8)
        plt.legend(['ZgH distribution', f'{name} distribution'], loc='upper left')
        plt.savefig(f'result/regression/AR_train/instance_normalization_{name}_vs_gt_distribution')
        plt.show()

def test_forward_data():
    global encoder
    global ztozg
    global generator

    encoder = normal_encoder()
    ztozg = ZtoZg()
    generator = generator()

    encoder.load_weights('model_weight/AE_encoder')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    generator.load_weights('model_weight/zd_zg_distillation_generator')

    path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
    path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

    ID = [f'ID{i}' for i in range(1, 91)]
    for id in ID:
        for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
            if count > 20:
                image = cv2.imread(path_AR_syn_train +  id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                z1, z2, z3 = encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
                zg1, _, _ = ztozg(z1)
                zg2, _, _ = ztozg(z2)
                zg3, _, _ = ztozg(z3)
                syn1, syn2, syn3 = generator(zg1), generator(zg2), generator(zg3)
                cv2.imwrite(f'cls_data/test_data_forward/{id}/{filename[0:-4]}-2-ratio-syn.jpg', np.array(tf.reshape(syn1, [64, 64]))*255)
                cv2.imwrite(f'cls_data/test_data_forward/{id}/{filename[0:-4]}-4-ratio-syn.jpg', np.array(tf.reshape(syn2, [64, 64]))*255)
                cv2.imwrite(f'cls_data/test_data_forward/{id}/{filename[0:-4]}-8-ratio-syn.jpg', np.array(tf.reshape(syn3, [64, 64]))*255)
                cv2.imwrite(f'cls_data/test_data_low_resolution/{id}/{filename[0:-4]}-2-ratio-low-resolution.jpg', np.array(tf.reshape(low1_image, [64, 64])) * 255)
                cv2.imwrite(f'cls_data/test_data_low_resolution/{id}/{filename[0:-4]}-4-ratio-low-resolution.jpg', np.array(tf.reshape(low2_image , [64, 64])) * 255)
                cv2.imwrite(f'cls_data/test_data_low_resolution/{id}/{filename[0:-4]}-8-ratio-low-resolution.jpg', np.array(tf.reshape(low3_image , [64, 64])) * 255)

    ID = [f'ID0{i}' if i < 10 else f'ID{i}' for i in range(1, 22)]
    for id in ID:
        for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
            image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            z1, z2, z3 = encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
            zg1, _, _ = ztozg(z1)
            zg2, _, _ = ztozg(z2)
            zg3, _, _ = ztozg(z3)
            syn1, syn2, syn3 = generator(zg1), generator(zg2), generator(zg3)
            cv2.imwrite(f'cls_data/test_data_forward/ID{int(id[2:]) + 90}/{filename[0:-4]}-2-ratio-syn.jpg', np.array(tf.reshape(syn1, [64, 64])) * 255)
            cv2.imwrite(f'cls_data/test_data_forward/ID{int(id[2:]) + 90}/{filename[0:-4]}-4-ratio-syn.jpg',  np.array(tf.reshape(syn2, [64, 64])) * 255)
            cv2.imwrite(f'cls_data/test_data_forward/ID{int(id[2:]) + 90}/{filename[0:-4]}-8-ratio-syn.jpg', np.array(tf.reshape(syn3, [64, 64])) * 255)
            cv2.imwrite(f'cls_data/test_data_low_resolution/ID{int(id[2:]) + 90}/{filename[0:-4]}-2-ratio-low-resolution.jpg', np.array(tf.reshape(low1_image, [64, 64])) * 255)
            cv2.imwrite(f'cls_data/test_data_low_resolution/ID{int(id[2:]) + 90}/{filename[0:-4]}-4-ratio-low-resolution.jpg', np.array(tf.reshape(low2_image, [64, 64])) * 255)
            cv2.imwrite(f'cls_data/test_data_low_resolution/ID{int(id[2:]) + 90}/{filename[0:-4]}-8-ratio-low-resolution.jpg', np.array(tf.reshape(low3_image, [64, 64])) * 255)

def regression_test_reg_loss():
    global encoder
    global ztozg
    global generator


    encoder = normal_encoder()
    ztozg = ZtoZg()
    ztozd = ZtoZd()
    reg = regression_model_with_instance()
    generator = generator()

    encoder.load_weights('model_weight/AE_encoder')
    ztozd.load_weights('model_weight/AE_ztozd')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    reg.load_weights('model_weight/regression_one_to_more')
    generator.load_weights('model_weight/zd_zg_distillation_generator')

    image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_train/ID1/43_train.jpg', 0) / 255
    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

    z1, z2, z3 = encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
    zd1, _ = ztozd(z1)
    zd2, _ = ztozd(z2)
    zd3, _ = ztozd(z3)

    zg1, _, _ = ztozg(z1)
    zg2, _, _ = ztozg(z2)
    zg3, _, _ = ztozg(z3)

    zreg1, zreg2, zreg3 = reg([zg1, zd1]), reg([zg2, zd2]), reg([zg3, zd3])

    corresponding_code = []
    for count, filename in enumerate(os.listdir('/home/bosen/PycharmProjects/Datasets/AR_train/ID1/')):
        if count < 20:
            print(filename)
            image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_train/ID1/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            z1 = encoder(image.reshape(1, 64, 64, 1))
            zg1, _, _ = ztozg(z1)
            corresponding_code.append(tf.reshape(zg1, [200]))
    corresponding_code = np.array(corresponding_code)

    zreg1_error = tf.reduce_mean(tf.square(tf.tile(tf.reshape(zreg1, [1, 200]), [corresponding_code.shape[0], 1]) - corresponding_code), axis=-1)
    zreg2_error = tf.reduce_mean(tf.square(tf.tile(tf.reshape(zreg2, [1, 200]), [corresponding_code.shape[0], 1]) - corresponding_code), axis=-1)
    zreg3_error = tf.reduce_mean(tf.square(tf.tile(tf.reshape(zreg3, [1, 200]), [corresponding_code.shape[0], 1]) - corresponding_code), axis=-1)
    print(zreg1_error)
    print(zreg2_error)
    print(zreg3_error)
    plt.plot(zreg1_error)
    plt.show()
    plt.plot(zreg2_error)
    plt.show()
    plt.plot(zreg3_error)
    plt.show()

def regression_test_condition():
    global encoder
    global ztozg
    global generator

    encoder = normal_encoder()
    ztozg = ZtoZg()
    ztozd = ZtoZd()
    reg = regression_model_with_instance()
    generator = generator()

    encoder.load_weights('model_weight/AE_encoder')
    ztozd.load_weights('model_weight/AE_ztozd')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    reg.load_weights('model_weight/regression_one_to_one3')
    generator.load_weights('model_weight/zd_zg_distillation_generator')

    image = cv2.imread('/home/bosen/PycharmProjects/Datasets/AR_test/ID21/11_test.jpg', 0) / 255
    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

    z0, z1, z2, z3 = encoder(image.reshape(1, 64, 64, 1)), encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
    zd0, _ = ztozd(z0)
    zd1, _ = ztozd(z1)
    zd2, _ = ztozd(z2)
    zd3, _ = ztozd(z3)

    zg0, _, _ = ztozg(z0)
    zg1, _, _ = ztozg(z1)
    zg2, _, _ = ztozg(z2)
    zg3, _, _ = ztozg(z3)

    zd1_zero = tf.zeros((1, 200))
    zd2_zero = tf.zeros((1, 200))
    zd3_zero = tf.zeros((1, 200))

    zd1_one = tf.ones((1, 200))
    zd2_one = tf.ones((1, 200))
    zd3_one = tf.ones((1, 200))


    zreg0_1, zreg1_1, zreg2_1, zreg3_1 = reg([zg0, zd0]), reg([zg1, zd1]), reg([zg2, zd2]), reg([zg3, zd3])
    zreg0_2, zreg1_2, zreg2_2, zreg3_2 = reg([zreg0_1, zd1]), reg([zreg1_1, zd1]), reg([zreg2_1, zd2]), reg([zreg3_1, zd3])
    zreg0_3, zreg1_3, zreg2_3, zreg3_3 = reg([zreg0_2, zd1]), reg([zreg1_2, zd1]), reg([zreg2_2, zd2]), reg([zreg3_2, zd3])

    zreg0_zero, zreg1_zero, zreg2_zero, zreg3_zero = reg([zg0, zd1_zero]), reg([zg1, zd1_zero]), reg([zg2, zd2_zero]), reg([zg3, zd3_zero])
    zreg0_one, zreg1_one, zreg2_one, zreg3_one = reg([zg0, zd1_zero]), reg([zg1, zd1_one]), reg([zg2, zd2_one]), reg([zg3, zd3_one])


    syn0_reg_1, syn1_reg_1, syn2_reg_1, syn3_reg_1 = generator(zreg0_1), generator(zreg1_1), generator(zreg2_1), generator(zreg3_1)
    syn0_reg_2, syn1_reg_2, syn2_reg_2, syn3_reg_2 = generator(zreg0_2), generator(zreg1_2), generator(zreg2_2), generator(zreg3_2)
    syn0_reg_3, syn1_reg_3, syn2_reg_3, syn3_reg_3 = generator(zreg0_3), generator(zreg1_3), generator(zreg2_3), generator(zreg3_3)
    syn0_reg_zero, syn1_reg_zero, syn2_reg_zero, syn3_reg_zero = generator(zreg0_zero), generator(zreg1_zero), generator(zreg2_zero), generator(zreg3_zero)
    syn0_reg_one, syn1_reg_one, syn2_reg_one, syn3_reg_one = generator(zreg0_one), generator(zreg1_one), generator(zreg2_one), generator(zreg3_one)
    for_syn0, for_syn1, for_syn2, for_syn3 = generator(zg0), generator(zg1), generator(zg2), generator(zg3)

    plt.subplots(figsize=(4, 7))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(7, 4, 1)
    plt.axis('off')
    plt.imshow(image.reshape(64, 64), cmap='gray')
    plt.subplot(7, 4, 2)
    plt.axis('off')
    plt.imshow(low1_image.reshape(64, 64), cmap='gray')
    plt.subplot(7, 4, 3)
    plt.axis('off')
    plt.imshow(low2_image.reshape(64, 64), cmap='gray')
    plt.subplot(7, 4, 4)
    plt.axis('off')
    plt.imshow(low3_image.reshape(64, 64), cmap='gray')

    plt.subplot(7, 4, 5)
    plt.axis('off')
    plt.imshow(tf.reshape(for_syn0, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 6)
    plt.axis('off')
    plt.imshow(tf.reshape(for_syn1, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 7)
    plt.axis('off')
    plt.imshow(tf.reshape(for_syn2, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 8)
    plt.axis('off')
    plt.imshow(tf.reshape(for_syn3, [64, 64]), cmap='gray')

    plt.subplot(7, 4, 9)
    plt.axis('off')
    plt.imshow(tf.reshape(syn0_reg_1, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 10)
    plt.axis('off')
    plt.imshow(tf.reshape(syn1_reg_1, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 11)
    plt.axis('off')
    plt.imshow(tf.reshape(syn2_reg_1, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 12)
    plt.axis('off')
    plt.imshow(tf.reshape(syn3_reg_1, [64, 64]), cmap='gray')

    plt.subplot(7, 4, 13)
    plt.axis('off')
    plt.imshow(tf.reshape(syn0_reg_2, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 14)
    plt.axis('off')
    plt.imshow(tf.reshape(syn1_reg_2, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 15)
    plt.axis('off')
    plt.imshow(tf.reshape(syn2_reg_2, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 16)
    plt.axis('off')
    plt.imshow(tf.reshape(syn3_reg_2, [64, 64]), cmap='gray')

    plt.subplot(7, 4, 17)
    plt.axis('off')
    plt.imshow(tf.reshape(syn0_reg_3, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 18)
    plt.axis('off')
    plt.imshow(tf.reshape(syn1_reg_3, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 19)
    plt.axis('off')
    plt.imshow(tf.reshape(syn2_reg_3, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 20)
    plt.axis('off')
    plt.imshow(tf.reshape(syn3_reg_3, [64, 64]), cmap='gray')

    plt.subplot(7, 4, 21)
    plt.axis('off')
    plt.imshow(tf.reshape(syn0_reg_one, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 22)
    plt.axis('off')
    plt.imshow(tf.reshape(syn1_reg_one, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 23)
    plt.axis('off')
    plt.imshow(tf.reshape(syn2_reg_one, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 24)
    plt.axis('off')
    plt.imshow(tf.reshape(syn3_reg_one, [64, 64]), cmap='gray')

    plt.subplot(7, 4, 25)
    plt.axis('off')
    plt.imshow(tf.reshape(syn0_reg_zero, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 26)
    plt.axis('off')
    plt.imshow(tf.reshape(syn1_reg_zero, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 27)
    plt.axis('off')
    plt.imshow(tf.reshape(syn2_reg_zero, [64, 64]), cmap='gray')
    plt.subplot(7, 4, 28)
    plt.axis('off')
    plt.imshow(tf.reshape(syn3_reg_zero, [64, 64]), cmap='gray')
    plt.show()
    plt.close()

    psnr_reg_1, psnr_reg_2, psnr_reg_3, psnr_for, psnr_one, psnr_zero = [], [], [], [], [], []
    ssim_reg_1, ssim_reg_2, ssim_reg_3, ssim_for, ssim_one, ssim_zero = [], [], [], [], [], []

    psnr_reg_1.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_1, max_val=1)[0])
    psnr_reg_1.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_1, max_val=1)[0])
    psnr_reg_1.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_1, max_val=1)[0])
    psnr_reg_1.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_1, max_val=1)[0])

    psnr_reg_2.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_2, max_val=1)[0])
    psnr_reg_2.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_2, max_val=1)[0])
    psnr_reg_2.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_2, max_val=1)[0])
    psnr_reg_2.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_2, max_val=1)[0])

    psnr_reg_3.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_3, max_val=1)[0])
    psnr_reg_3.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_3, max_val=1)[0])
    psnr_reg_3.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_3, max_val=1)[0])
    psnr_reg_3.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_3, max_val=1)[0])

    psnr_one.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_one, max_val=1)[0])
    psnr_one.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_one, max_val=1)[0])
    psnr_one.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_one, max_val=1)[0])
    psnr_one.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_one, max_val=1)[0])

    psnr_zero.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_zero, max_val=1)[0])
    psnr_zero.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_zero, max_val=1)[0])
    psnr_zero.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_zero, max_val=1)[0])
    psnr_zero.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_zero, max_val=1)[0])

    psnr_for.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn0, max_val=1)[0])
    psnr_for.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn1, max_val=1)[0])
    psnr_for.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn2, max_val=1)[0])
    psnr_for.append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn3, max_val=1)[0])

    ssim_reg_1.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_1, max_val=1)[0])
    ssim_reg_1.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_1, max_val=1)[0])
    ssim_reg_1.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_1, max_val=1)[0])
    ssim_reg_1.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_1, max_val=1)[0])

    ssim_reg_2.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_2, max_val=1)[0])
    ssim_reg_2.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_2, max_val=1)[0])
    ssim_reg_2.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_2, max_val=1)[0])
    ssim_reg_2.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_2, max_val=1)[0])

    ssim_reg_3.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_3, max_val=1)[0])
    ssim_reg_3.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_3, max_val=1)[0])
    ssim_reg_3.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_3, max_val=1)[0])
    ssim_reg_3.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_3, max_val=1)[0])

    ssim_one.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_one, max_val=1)[0])
    ssim_one.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_one, max_val=1)[0])
    ssim_one.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_one, max_val=1)[0])
    ssim_one.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_one, max_val=1)[0])

    ssim_zero.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn0_reg_zero, max_val=1)[0])
    ssim_zero.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn1_reg_zero, max_val=1)[0])
    ssim_zero.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn2_reg_zero, max_val=1)[0])
    ssim_zero.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), syn3_reg_zero, max_val=1)[0])

    ssim_for.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn0, max_val=1)[0])
    ssim_for.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn1, max_val=1)[0])
    ssim_for.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn2, max_val=1)[0])
    ssim_for.append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), for_syn3, max_val=1)[0])

    plt.plot(psnr_reg_1)
    plt.plot(psnr_reg_2)
    plt.plot(psnr_reg_3)
    plt.plot(psnr_one)
    plt.plot(psnr_zero)
    plt.plot(psnr_for)
    plt.legend(['reg_1_times', 'reg_2_times', 'reg_3_times', 'reg_1_times_condition=0', 'reg_1_times_condition=1', 'forward'])
    plt.xticks([0, 1, 2, 3], ['self', '2-ratio', '4-ratio', '8-ratio'])
    plt.show()

    plt.plot(ssim_reg_1)
    plt.plot(ssim_reg_2)
    plt.plot(ssim_reg_3)
    plt.plot(ssim_one)
    plt.plot(ssim_zero)
    plt.plot(ssim_for)
    plt.legend(['reg_1_times', 'reg_2_times', 'reg_3_times', 'reg_1_times_condition=0', 'reg_1_times_condition=1', 'forward'])
    plt.xticks([0, 1, 2, 3], ['self', '2-ratio', '4-ratio', '8-ratio'])
    plt.show()




if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    # compare_with_distllation_or_without_distillation_forward_answer()
    inversion_vs_forward_mpsnr_mssim()








