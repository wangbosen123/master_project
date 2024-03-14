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

class reid_image_classifier2():
    def __init__(self, epochs, batch_num, batch_size):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.opti = tf.keras.optimizers.Adam(1e-4)
        self.cls = classifier()
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.regression = regression_model()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        # self.cls.load_weights('model_weight/cls_aug')
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
        # self.regression.load_weights('model_weight/regression')
        self.generator.load_weights('model_weight/zd_zg_distillation_generator')

        self.last_conv_layer_name = 'conv2d'
        self.network_layer_name = ['max_pooling2d', 'batch_normalization', 'conv2d_1', 'max_pooling2d_1',
                                   'batch_normalization_1', 'conv2d_2',
                                   'max_pooling2d_2', 'batch_normalization_2', 'flatten', 'dense_13', 'dropout',
                                   'dense_14']

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
                    #     train_label.append(tf.one_hot(int(id[2:]) - 1, 111))
                    train_label.append(tf.one_hot(int(id[2:]) - 1, 111))
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
            _, attention_map = self.gradcam_heatmap_mutiple(image, self.cls, self.last_conv_layer_name,
                                                            self.network_layer_name, label, corresponding_label=True)
            inverse_image = image * (tf.sigmoid(1 - attention_map))

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

            if epoch % 10 == 0:
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
                    label.append(int(id[2:]) - 1)
                    image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    low1_test = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64),
                                           cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_test = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64),
                                           cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_test = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64),
                                           cv2.INTER_CUBIC).reshape(1, 64, 64, 1)

                    z1, z2, z3 = self.encoder(low1_test), self.encoder(low2_test), self.encoder(low3_test)
                    zg1, _, _ = self.ztozg(z1)
                    zg2, _, _ = self.ztozg(z2)
                    zg3, _, _ = self.ztozg(z3)

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

        confusion_matrix = [confusion_matrix_low1, confusion_matrix_low2, confusion_matrix_low3,
                            confusion_matrix_forward_low1, confusion_matrix_forward_low2, confusion_matrix_forward_low3]
        accuracy = [accuracy_low1, accuracy_low2, accuracy_low3, accuracy_forward_low1, accuracy_forward_low2,
                    accuracy_forward_low3]

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


if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    cls = reid_image_classifier2(20, batch_size=20, batch_num=111)
    cls.main(reid=False)
