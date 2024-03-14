import os
import tensorflow as tf
from build_model import *
from test import *
import numpy as np
import cv2
import time
from sklearn.metrics import accuracy_score
import csv
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

class face_recognition_training():
    def __init__(self, epochs, batch_num, batch_size):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        self.encoder = encoder()
        self.ztozd = ZtoZd()
        self.ZtoZg = ZtoZg()
        self.H_cls = H_cls_no_condition()
        self.generator = generator()

        self.encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation')
        self.ztozd.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
        self.ZtoZg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
        self.generator.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation')
        self.opti = tf.keras.optimizers.Adam(1e-3)
        self.ar_train_path, self.ar_label = self.prepare_data()
        self.features_zd, self.features_zg, self.db_label = self.get_database_feature()
        print(self.ar_train_path.shape, self.ar_label.shape)
        print(self.features_zd.shape, self.features_zg.shape, self.db_label.shape)

    def prepare_data(self):
        path_AR_train = 'cls_datasets/train_data_var_less/'
        ar_train_path = []
        ar_train_label = []

        for ID in os.listdir(path_AR_train):
            for num, filename in enumerate(os.listdir(path_AR_train + ID)):
                ar_train_path.append(path_AR_train + ID + '/' + filename)
                ar_train_label.append(tf.one_hot(int(ID[2:])-1, 111))

        ar_data = list(zip(ar_train_path, ar_train_label))
        np.random.shuffle(ar_data)
        ar_train_data = list(zip(*ar_data))
        return np.array(ar_train_data[0]), np.array(ar_train_data[1]),

    def get_database_feature(self):
        path_AR_train = 'cls_datasets/train_data_var_less/'
        features_zd = []
        features_zg = []
        features_label = []

        for ID in os.listdir(path_AR_train):
            for num, filename in enumerate(os.listdir(path_AR_train + ID)):
                image = cv2.imread(path_AR_train + ID + '/' + filename, 0) / 255
                image = image.reshape(1, 64, 64, 1)
                z, _ = self.encoder(image)
                feature_zd, _ = self.ztozd(z)
                feature_zg, _, _ = self.ZtoZg(z)
                features_zd.append(tf.reshape(feature_zd, [200]))
                features_zg.append(tf.reshape(feature_zg, [200]))
                features_label.append(tf.one_hot(int(ID[2:])-1, 111))
        features_zd, features_zg, features_label = np.array(features_zd), np.array(features_zg), np.array(features_label)
        return features_zd, features_zg, features_label

    def find_neighbor(self, batch_features_zd, batch_features_zg, train=True):
        data = []
        for num, feature in enumerate(batch_features_zg):
            # total_feature_zd = tf.tile(tf.reshape(feature, [1, 200]), [self.features_zd.shape[0], 1])
            total_feature_zg = tf.tile(tf.reshape(feature, [1, 200]), [self.features_zg.shape[0], 1])
            # similarity_zd = list(cosine_similarity(self.features_zd, total_feature_zd)[:, 0]+1)
            similarity_zg = list(cosine_similarity(self.features_zg, total_feature_zg)[:, 0]+1)
            if train:
                similarity_zg[similarity_zg.index(max(similarity_zg))] = 0

            num_neighbor = 0
            neighbor = []
            neighbor_label = []
            similarity_score_zg = []
            similarity_score_zd = []
            while num_neighbor < 3:
                if len(neighbor) > 0:
                    if np.argmax(self.db_label[similarity_zg.index(max(similarity_zg))], axis=-1) in neighbor_label:
                        similarity_zg[similarity_zg.index(max(similarity_zg))] = 0
                    else:
                        neighbor.append(self.features_zg[similarity_zg.index(max(similarity_zg))])
                        neighbor_label.append(np.argmax(self.db_label[similarity_zg.index(max(similarity_zg))], axis=-1))
                        # similarity_score_zd.append(similarity_zd[similarity_zd.index(max(similarity_zd))])
                        similarity_score_zg.append(similarity_zg[similarity_zg.index(max(similarity_zg))])
                        num_neighbor += 1
                else:
                    neighbor.append(self.features_zg[similarity_zg.index(max(similarity_zg))])
                    neighbor_label.append(np.argmax(self.db_label[similarity_zg.index(max(similarity_zg))], axis=-1))
                    # similarity_score_zd.append(similarity_zd[similarity_zd.index(max(similarity_zd))])
                    similarity_score_zg.append(similarity_zg[similarity_zg.index(max(similarity_zg))])
                    num_neighbor += 1
            neighbor_1 = (similarity_score_zg[0]/np.sum(similarity_score_zg)) * neighbor[0]
            neighbor_2 = (similarity_score_zg[1]/np.sum(similarity_score_zg)) * neighbor[1]
            neighbor_3 = (similarity_score_zg[2]/np.sum(similarity_score_zg)) * neighbor[2]

            feature = tf.concat([tf.reshape(feature, [1, 200]), tf.reshape(neighbor_1, [1, 200])], axis=0)
            feature = tf.concat([feature, tf.reshape(neighbor_2, [1, 200])], axis=0)
            feature = tf.concat([feature, tf.reshape(neighbor_3, [1, 200])], axis=0)
            data.append(feature)
        data = np.array(data)
        return data

    def get_batch_data(self, data, batch_idx, batch_size, image=True):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        if image:
            images = []
            for path in train_data:
                image = cv2.imread(path, 0) / 255
                images.append(image)

            images = np.array(images).reshape(-1, 64, 64, 1)
            return images
        else:
            return np.array(train_data)

    def train_step(self, real_image, label, train=True):
        with tf.GradientTape() as tape:
            z, _ = self.encoder(real_image)
            features_zd, _ = self.ztozd(z)
            features_zg, _, _ = self.ZtoZg(z)
            # features_zg = self.find_neighbor(features_zd, features_zg)
            _, pred = self.H_cls(features_zg)

            cce = tf.keras.losses.CategoricalCrossentropy()
            ce_loss = cce(label, pred)
            acc = accuracy_score(np.argmax(pred, axis=-1), np.argmax(label, axis=-1))
            total_loss = ce_loss
        if train:
            grads = tape.gradient(total_loss, self.H_cls.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.H_cls.trainable_variables))
        return ce_loss, acc

    def training(self, training=True):
        ce_loss_epoch = []
        acc_epoch = []

        for epoch in range(1, self.epochs + 1):
            ce_loss_batch = []
            acc_batch = []

            start = time.time()
            for step in range(self.batch_num):
                real_image = self.get_batch_data(self.ar_train_path, step, self.batch_size)
                ar_label = self.get_batch_data(self.ar_label, step, self.batch_size, image=False)
                ce_loss, acc = self.train_step(real_image, ar_label, train=training)

                ce_loss_batch.append(ce_loss)
                acc_batch.append(acc)
            ce_loss_epoch.append(np.mean(ce_loss_batch))
            acc_epoch.append(np.mean(acc_batch))
            print('Start of epoch %d' % (epoch,))
            print(f'the ce_los is {ce_loss_epoch[-1]}')
            print(f'the accuracy is {acc_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('-----------------------------------------------')
            # self.encoder.save_weights('model_weight/encoder_init_c')
            # self.ZtoZg.save_weights('model_weight/ztozg_init_c')
            self.H_cls.save_weights('model_weight/H.cls2')
        # plt.plot(ce_loss_epoch)
        # plt.xlabel('epoch')
        # plt.ylabel('loss value')
        # plt.savefig('model_weight/model_ce_loss')
        # plt.close()

    def recognition(self, condition):
        path_AR_test = 'cls_datasets/train_data_var_large/'
        if condition:
            cls = H_cls()
            cls.load_weights('model_weight/H.cls2_condition')
        if not condition:
            cls = H_cls_no_condition()
            cls.load_weights('model_weight/H.cls2')

        number = 0
        rank_32, rank_20, rank_16, rank_12, rank_8 = [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)]
        for ID in os.listdir(path_AR_test):
            for num, filename in enumerate(os.listdir(path_AR_test + ID)):
                if '-3-0' in filename and int(ID[2:]) < 91:
                    number += 1
                    print(int(ID[2:]))
                    image = cv2.imread(path_AR_test + ID + '/' + filename, 0) / 255
                    low1_image = cv2.resize(image, (32, 32), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(image, (20, 20), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(image, (16, 16), cv2.INTER_CUBIC)
                    low4_image = cv2.resize(image, (12, 12), cv2.INTER_CUBIC)
                    low5_image = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
                    low1_image = cv2.resize(low1_image, (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(low2_image, (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(low3_image, (64, 64), cv2.INTER_CUBIC)
                    low4_image = cv2.resize(low4_image, (64, 64), cv2.INTER_CUBIC)
                    low5_image = cv2.resize(low5_image, (64, 64), cv2.INTER_CUBIC)

                    z32, _ = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    z20, _ = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    z16, _ = self.encoder(low3_image.reshape(1, 64, 64, 1))
                    z12, _ = self.encoder(low4_image.reshape(1, 64, 64, 1))
                    z8, _ = self.encoder(low5_image.reshape(1, 64, 64, 1))


                    feature_32d, _ = self.ztozd(z32)
                    feature_20d, _ = self.ztozd(z20)
                    feature_16d, _ = self.ztozd(z16)
                    feature_12d, _ = self.ztozd(z12)
                    feature_8d, _ = self.ztozd(z8)

                    feature_32g, _, _ = self.ZtoZg(z32)
                    feature_20g, _, _ = self.ZtoZg(z20)
                    feature_16g, _, _ = self.ZtoZg(z16)
                    feature_12g, _, _ = self.ZtoZg(z12)
                    feature_8g, _, _ = self.ZtoZg(z8)

                    if condition:
                        feature_32g = self.find_neighbor(feature_32d, feature_32g, train=False)
                        feature_20g = self.find_neighbor(feature_20d, feature_20g, train=False)
                        feature_16g = self.find_neighbor(feature_16d, feature_16g, train=False)
                        feature_12g = self.find_neighbor(feature_12d, feature_12g, train=False)
                        feature_8g = self.find_neighbor(feature_8d, feature_8g, train=False)

                    _, pred_32g = cls(feature_32g)
                    _, pred_20g = cls(feature_20g)
                    _, pred_16g = cls(feature_16g)
                    _, pred_12g = cls(feature_12g)
                    _, pred_8g = cls(feature_8g)


                    res_pred_32g, res_pred_20g, res_pred_16g, res_pred_12g, res_pred_8g = pred_32g.numpy(), pred_20g.numpy(), pred_16g.numpy(), pred_12g.numpy(), pred_8g.numpy()
                    for i in range(10):
                        if np.argmax(res_pred_32g, axis=-1) == np.argmax(tf.one_hot(int(ID[2:])-1, 111), axis=-1):
                            for j in range(i, 10):
                                rank_32[j] += 1
                            break
                        else:
                            res_pred_32g[0][np.argmax(res_pred_32g)] = 0

                    for i in range(10):
                        if np.argmax(res_pred_20g, axis=-1) == np.argmax(tf.one_hot(int(ID[2:])-1, 111), axis=-1):
                            for j in range(i, 10):
                                rank_20[j] += 1
                            break
                        else:
                            res_pred_20g[0][np.argmax(res_pred_20g)] = 0

                    for i in range(10):
                        if np.argmax(res_pred_16g, axis=-1) == np.argmax(tf.one_hot(int(ID[2:])-1, 111), axis=-1):
                            for j in range(i, 10):
                                rank_16[j] += 1
                            break
                        else:
                            res_pred_16g[0][np.argmax(res_pred_16g)] = 0

                    for i in range(10):
                        if np.argmax(res_pred_12g, axis=-1) == np.argmax(tf.one_hot(int(ID[2:])-1, 111), axis=-1):
                            for j in range(i, 10):
                                rank_12[j] += 1
                            break
                        else:
                            res_pred_12g[0][np.argmax(res_pred_12g)] = 0

                    for i in range(10):
                        if np.argmax(res_pred_8g, axis=-1) == np.argmax(tf.one_hot(int(ID[2:])-1, 111), axis=-1):
                            for j in range(i, 10):
                                rank_8[j] += 1
                            break
                        else:
                            res_pred_8g[0][np.argmax(res_pred_8g)] = 0


        rank_32_syn, rank_20_syn, rank_16_syn, rank_12_syn, rank_8_syn = [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)]
        for id in os.listdir(path_AR_test):
            for num, filename in enumerate(os.listdir(path_AR_test + id)):
                if 'syn' in filename and int(id[2:]) < 91:
                    image = cv2.imread(path_AR_test + '/' + id + '/' + filename, 0)/255
                    z, _ = self.encoder(image.reshape(1, 64, 64, 1))
                    feature_zd, _ = self.ztozd(z)
                    feature_zg, _, _ = self.ZtoZg(z)
                    if condition:
                        feature_zg = self.find_neighbor(feature_zd, feature_zg, train=False)
                    _, pred = cls(feature_zg)

                    res_pred = pred.numpy()
                    if '32' in filename:
                        for i in range(10):
                            if np.argmax(res_pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:])-1, 111), axis=-1):
                                for j in range(i, 10):
                                    rank_32_syn[j] += 1
                                break
                            else:
                                res_pred[0][np.argmax(res_pred)] = 0

                    if '20' in filename:
                        for i in range(10):
                            if np.argmax(res_pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:])-1, 111), axis=-1):
                                for j in range(i, 10):
                                    rank_20_syn[j] += 1
                                break
                            else:
                                res_pred[0][np.argmax(res_pred)] = 0

                    if '16' in filename:
                        for i in range(10):
                            if np.argmax(res_pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:])-1, 111), axis=-1):
                                for j in range(i, 10):
                                    rank_16_syn[j] += 1
                                break
                            else:
                                res_pred[0][np.argmax(res_pred)] = 0

                    if '12' in filename:
                        for i in range(10):
                            if np.argmax(res_pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:])-1, 111), axis=-1):
                                for j in range(i, 10):
                                    rank_12_syn[j] += 1
                                break
                            else:
                                res_pred[0][np.argmax(res_pred)] = 0

                    if '8' in filename:
                        for i in range(10):
                            if np.argmax(res_pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:])-1, 111), axis=-1):
                                for j in range(i, 10):
                                    rank_8_syn[j] += 1
                                break
                            else:
                                res_pred[0][np.argmax(res_pred)] = 0

        rank_32 = (np.array(rank_32)) / number
        rank_20 = (np.array(rank_20)) / number
        rank_16 = (np.array(rank_16)) / number
        rank_12 = (np.array(rank_12)) / number
        rank_8 = (np.array(rank_8)) / number
        rank_l = (rank_32 + rank_20 + rank_16 + rank_12 + rank_8) / 5

        rank_32_syn = (np.array(rank_32_syn)) / number
        rank_20_syn = (np.array(rank_20_syn)) / number
        rank_16_syn = (np.array(rank_16_syn)) / number
        rank_12_syn = (np.array(rank_12_syn)) / number
        rank_8_syn = (np.array(rank_8_syn)) / number
        rank_syn = (rank_32_syn + rank_20_syn + rank_16_syn + rank_12_syn + rank_8_syn) / 5
        print(rank_32)
        print(rank_20)
        print(rank_16)
        print(rank_12)
        print(rank_8)
        print(rank_l)
        print('-------')
        print(rank_32_syn)
        print(rank_20_syn)
        print(rank_16_syn)
        print(rank_12_syn)
        print(rank_8_syn)
        print(rank_syn)
        # plt.plot(rank_l)
        # plt.plot(rank_syn)
        # plt.legend(['L', 'Syn'], loc='lower right')
        # plt.title('Rank-10-Acc')
        # plt.show()
        # plt.close()

        plt.plot(rank_32, linestyle="-.", color='blue', marker='.')
        plt.plot(rank_32_syn, color='blue', marker='.')
        plt.plot(rank_20, linestyle="-.", color='red', marker='.')
        plt.plot(rank_20_syn, color='red', marker='.')
        plt.plot(rank_16, linestyle="-.", color='y', marker='.')
        plt.plot(rank_16_syn, color='y', marker='.')
        plt.plot(rank_12, linestyle="-.", color='c', marker='.')
        plt.plot(rank_12_syn, color='c', marker='.')
        plt.plot(rank_8, linestyle="-.", color='k', marker='.')
        plt.plot(rank_8_syn, color='k', marker='.')
        plt.legend(['L32', 'L32-syn', 'L20', 'L20-syn', "L16", 'L16-syn', 'L12', 'L12_syn', 'L8', 'L8_syn'], loc='lower right')
        plt.title('Rank-10-Acc')
        plt.show()
        plt.close()
        return rank_l, rank_syn

    def compare_rank(self):
        rank_l, rank_syn = self.recognition(condition=False)
        rank_l_condition, rank_syn_condition = self.recognition(condition=True)
        plt.plot(rank_l, linestyle="-.", color='blue', marker='.')
        plt.plot(rank_syn, linestyle="-.", color='red', marker='.')
        plt.plot(rank_l_condition, color='blue', marker='.')
        plt.plot(rank_syn_condition, color='red', marker='.')
        plt.legend(['L', 'Syn', "L+condition", "Syn+condition"], loc='lower right')
        plt.show()
        plt.close()

    def experiment(self):
        train_path = 'cls_datasets/train_data_var_less/'
        test_path = 'cls_datasets/cls_test_data/'
        ids = ['ID91/', 'ID92/', 'ID93/', 'ID94/', 'ID95/', 'ID96/', 'ID110/', 'ID111/']
        zg_feature_train, zg_feature_test = [[] for i in range(8)], [[] for i in range(8)]
        zd_feature_train, zd_feature_test = [[] for i in range(8)], [[] for i in range(8)]

        for num, id in enumerate(ids):
            for filename in os.listdir(train_path + id):
                image = cv2.imread(train_path + id + '/' + filename, 0) / 255
                image = image.reshape(1, 64, 64, 1)
                z, _ = self.encoder(image)
                feature_zd, _ = self.ztozd(z)
                feature_zg, _, _ = self.ZtoZg(z)
                zd_feature_train[num].append(tf.reshape(feature_zd, [200]))
                zg_feature_train[num].append(tf.reshape(feature_zg, [200]))


        for num, id in enumerate(ids):
            for filename in os.listdir(test_path + id):
                if "syn" in filename:
                    image = cv2.imread(test_path + id + '/' + filename, 0) / 255
                    image = image.reshape(1, 64, 64, 1)
                    z, _ = self.encoder(image)
                    feature_zd, _ = self.ztozd(z)
                    feature_zg, _, _ = self.ZtoZg(z)
                    zd_feature_test[num].append(tf.reshape(feature_zd, [200]))
                    zg_feature_test[num].append(tf.reshape(feature_zg, [200]))

        zg_feature_train, zd_feature_train = np.array(zg_feature_train), np.array(zd_feature_train)
        zg_feature_test, zd_feature_test = np.array(zg_feature_test), np.array(zd_feature_test)
        print(zg_feature_train.shape)
        print(zg_feature_test.shape)

        def find_neighbor(batch_features_zg, batch_features_zd, train=True):
            neighbor_label = [[] for i in range(8)]
            for count, id_feature in enumerate(batch_features_zd):
                for num, feature in enumerate(id_feature):
                    total_feature_zd = tf.tile(tf.reshape(feature, [1, 200]), [self.features_zd.shape[0], 1])
                    total_feature_zg = tf.tile(tf.reshape(feature, [1, 200]), [self.features_zg.shape[0], 1])
                    similarity_zd = list(cosine_similarity(self.features_zd, total_feature_zd)[:, 0] + 1)
                    similarity_zg = list(cosine_similarity(self.features_zg, total_feature_zg)[:, 0] + 1)

                    if train:
                        similarity_zd[similarity_zd.index(max(similarity_zd))] = 0
                        similarity_zg[similarity_zg.index(max(similarity_zg))] = 0

                    num_neighbor = 0
                    res_neighbor_label = []
                    while num_neighbor < 4:
                        if len(neighbor_label[count]) > 0:
                            if np.argmax(self.db_label[similarity_zd.index(max(similarity_zd))], axis=-1) in res_neighbor_label:
                                similarity_zd[similarity_zd.index(max(similarity_zd))] = 0
                            else:
                                neighbor_label[count].append(np.argmax(self.db_label[similarity_zd.index(max(similarity_zd))], axis=-1))
                                res_neighbor_label.append(np.argmax(self.db_label[similarity_zd.index(max(similarity_zd))], axis=-1))
                                num_neighbor += 1
                        else:
                            neighbor_label[count].append(np.argmax(self.db_label[similarity_zd.index(max(similarity_zd))], axis=-1))
                            res_neighbor_label.append(np.argmax(self.db_label[similarity_zd.index(max(similarity_zd))], axis=-1))
                            num_neighbor += 1
            print(neighbor_label)
            return neighbor_label

        train_neighbor_label = find_neighbor(zg_feature_train, zd_feature_train, train=True)
        test_neighbor_label = find_neighbor(zg_feature_test, zd_feature_test, train=False)

        train_neighbor_id, test_neighbor_id = [[] for i in range(8)], [[] for i in range(8)]
        for num, id in enumerate(train_neighbor_label):
            for index, content in enumerate(id):
                if index == 0:
                    train_neighbor_id[num].append(content)
                else:
                    if content not in train_neighbor_id[num]:
                        train_neighbor_id[num].append(content)

        for num, id in enumerate(test_neighbor_label):
            for index, content in enumerate(id):
                if index == 0:
                    test_neighbor_id[num].append(content)
                else:
                    if content not in test_neighbor_id[num]:
                        test_neighbor_id[num].append(content)

        print(train_neighbor_id)
        print(test_neighbor_id)

        union, intersection = [0 for i in range(8)], [0 for i in range(8)]
        for num, id in enumerate(train_neighbor_id):
            for content in id:
                if content in test_neighbor_id[num]:
                    intersection[num] += 1
                else:
                    union[num] += 1
            union[num] += len(test_neighbor_id[num])
        print(union, intersection)

        x = [1, 2, 3, 4, 5, 6, 7, 8]
        x2 = [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8]
        plt.bar(x, union, color='b', width=0.4)
        plt.bar(x2, intersection, color='r', width=0.4, align='edge')
        plt.show()

def coeff_matrix():
    global encoder
    global ZtoZg
    global ZtoZd
    encoder = encoder()
    ztozd = ZtoZd()
    ztozg = ZtoZg()
    encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation')
    ztozd.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
    ztozg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')

    def get_database_feature():
        path_AR_train = 'cls_datasets/cls_train_data/'
        features_zd = []
        features_zg = []
        features_label = []
        for ID in os.listdir(path_AR_train):
            for num, filename in enumerate(os.listdir(path_AR_train + ID)):
                image = cv2.imread(path_AR_train + ID + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                image = image.reshape(1, 64, 64, 1)
                z, _ = encoder(image)
                feature_zd, _ = ztozd(z)
                feature_zg, _, _ = ztozg(z)
                features_zd.append(tf.reshape(feature_zd, [200]))
                features_zg.append(tf.reshape(feature_zg, [200]))
                features_label.append(tf.one_hot(int(ID[2:])-1, 111))
        features_zd, features_zg, features_label = np.array(features_zd), np.array(features_zg), np.array(features_label)
        return features_zd, features_zg, features_label
    features_zd, features_zg, features_label = get_database_feature()

    def find_neighbor(features):
        for num, feature in enumerate(features):
            total_feature_zd = tf.tile(tf.reshape(feature, [1, 200]), [features_zd.shape[0], 1])
            similarity_zd = list(cosine_similarity(features_zd, total_feature_zd)[:, 0])

            num_neighbor = 0
            neighbor = []
            neighbor_label = []
            similarity_score = []
            while num_neighbor < 4:
                if len(neighbor) > 0:
                    if np.argmax(features_label[similarity_zd.index(max(similarity_zd))], axis=-1)+1 in neighbor_label:
                        similarity_zd[similarity_zd.index(max(similarity_zd))] = 0
                    else:
                        neighbor.append(features_zd[similarity_zd.index(max(similarity_zd))])
                        neighbor_label.append(np.argmax(features_label[similarity_zd.index(max(similarity_zd))], axis=-1)+1)
                        similarity_score.append(similarity_zd[similarity_zd.index(max(similarity_zd))])
                        num_neighbor += 1
                else:
                    neighbor.append(features_zd[similarity_zd.index(max(similarity_zd))])
                    neighbor_label.append(np.argmax(features_label[similarity_zd.index(max(similarity_zd))], axis=-1)+1)
                    similarity_score.append(similarity_zd[similarity_zd.index(max(similarity_zd))])
                    num_neighbor += 1
            print(neighbor_label)
            print(similarity_score)
            print('-------')

    path = 'cls_datasets/cls_test_data/'
    ids = ['ID1/', 'ID2/', 'ID3/', 'ID4/', 'ID18/', 'ID19/', 'ID20/', 'ID21/']
    feat_zd, feat_zg = [], []
    for id in ids:
        for filename in os.listdir(path + id):
            if '-1-1' in filename:
                image = cv2.imread(path + id + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                low1_image = cv2.resize(image, (32, 32), cv2.INTER_CUBIC)
                low2_image = cv2.resize(image, (16, 16), cv2.INTER_CUBIC)
                low3_image = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
                low1_image = cv2.resize(low1_image, (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(low2_image, (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(low3_image, (64, 64), cv2.INTER_CUBIC)

                zh, _ = encoder(low1_image.reshape(1, 64, 64, 1))
                zm, _ = encoder(low2_image.reshape(1, 64, 64, 1))
                zl, _ = encoder(low3_image.reshape(1, 64, 64, 1))

                feature_zhd, _ = ztozd(zh)
                feature_zmd, _ = ztozd(zm)
                feature_zld, _ = ztozd(zl)
                feature_zh, _, _ = ztozg(zh)
                feature_zm, _, _ = ztozg(zm)
                feature_zl, _, _ = ztozg(zl)
                feat_zd.append(tf.reshape(feature_zhd, [200])), feat_zd.append(tf.reshape(feature_zmd, [200])), feat_zd.append(tf.reshape(feature_zld, [200]))
                feat_zg.append(tf.reshape(feature_zh, [200])), feat_zg.append(tf.reshape(feature_zm, [200])), feat_zg.append(tf.reshape(feature_zl, [200]))
    feat_zd, feat_zg = np.array(feat_zd), np.array(feat_zg)
    find_neighbor(feat_zd)

    aff_mtx_zd = tf.matmul(feat_zd, tf.transpose(feat_zd)) / (tf.matmul(tf.norm(feat_zd, axis=1, keepdims=True), tf.norm(tf.transpose(feat_zd), axis=0, keepdims=True)))
    H_trans_zd = aff_mtx_zd / tf.reshape(tf.reduce_sum(aff_mtx_zd, axis=1), [aff_mtx_zd.get_shape()[0], 1])

    aff_mtx_zg = tf.matmul(feat_zg, tf.transpose(feat_zg)) / (tf.matmul(tf.norm(feat_zg, axis=1, keepdims=True), tf.norm(tf.transpose(feat_zg), axis=0, keepdims=True)))
    H_trans_zg = aff_mtx_zg / tf.reshape(tf.reduce_sum(aff_mtx_zg, axis=1), [aff_mtx_zg.get_shape()[0], 1])

if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    face_recognition_training = face_recognition_training(epochs=6, batch_num=111, batch_size=6)
    # face_recognition_training.training()
    face_recognition_training.recognition(condition=False)
    # face_recognition_training.compare_rank()
    # face_recognition_training.experiment()

    # ids = ['ID92', 'ID93', 'ID94', 'ID109']
    # path = 'cls_datasets/train_data_var_less/'
    # plt.subplots(figsize=(6, 9))
    # plt.subplots_adjust(hspace=0, wspace=0)
    # for num, id in enumerate(ids):
    #     for filename in os.listdir(path + id):
    #         image = cv2.imread(path + id + '/' + filename, 0) / 255
    #         low1 = cv2.resize(image, (32, 32), cv2.INTER_CUBIC)
    #         low2 = cv2.resize(image, (20, 20), cv2.INTER_CUBIC)
    #         low3 = cv2.resize(image, (16, 16), cv2.INTER_CUBIC)
    #         low4 = cv2.resize(image, (12, 12), cv2.INTER_CUBIC)
    #         low5 = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
    #         low1 = cv2.resize(low1, (64, 64), cv2.INTER_CUBIC)
    #         low2 = cv2.resize(low2, (64, 64), cv2.INTER_CUBIC)
    #         low3 = cv2.resize(low3, (64, 64), cv2.INTER_CUBIC)
    #         low4 = cv2.resize(low4, (64, 64), cv2.INTER_CUBIC)
    #         low5 = cv2.resize(low5, (64, 64), cv2.INTER_CUBIC)
    #
    #         if '.bmp' in filename:
    #             plt.subplot(6, 4, num + 1)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    #             plt.subplot(6, 4, num + 5)
    #             plt.axis('off')
    #             plt.imshow(low1, cmap='gray')
    #             plt.subplot(6, 4, num + 9)
    #             plt.axis('off')
    #             plt.imshow(low2, cmap='gray')
    #             plt.subplot(6, 4, num + 13)
    #             plt.axis('off')
    #             plt.imshow(low3, cmap='gray')
    #             plt.subplot(6, 4, num + 17)
    #             plt.axis('off')
    #             plt.imshow(low4, cmap='gray')
    #             plt.subplot(6, 4, num + 21)
    #             plt.axis('off')
    #             plt.imshow(low5, cmap='gray')
    #         if '-3-0.bmp' in filename:
    #             plt.subplot(6, 4, num + 1)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    #         if "32_" in filename:
    #             plt.subplot(6, 4, num+5)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    #         if "20_" in filename:
    #             plt.subplot(6, 4, num+9)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    #         if "16_" in filename:
    #             plt.subplot(6, 4, num+13)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    #         if "12_" in filename:
    #             plt.subplot(6, 4, num+17)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    #         if "8_" in filename:
    #             plt.subplot(6, 4, num+21)
    #             plt.axis('off')
    #             plt.imshow(image, cmap='gray')
    # plt.show()















