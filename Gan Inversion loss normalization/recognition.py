from face_recognition import *

def prepare_data():
    data_path, data_label = [], []
    mean_image = []
    path = 'cls_datasets/train_data_var_less/'
    ID = [f'ID{i}' for i in range(1, 112)]

    for id in ID:
        res_images = 0
        number = 0
        for filename in os.listdir(path + id):
            data_path.append(path + id + '/' + filename)
            data_label.append(int(id[2:])-1)
            image = cv2.imread(path + id + '/' + filename, 0) / 255
            res_images += image
            number += 1
        res_images /= number
        mean_image.append(res_images)
    data = list(zip(data_path, data_label))
    np.random.shuffle(data)
    data = list(zip(*data))
    data_path, data_label, mean_image = np.array(data[0]), np.array(data[1]), np.array(mean_image)
    return data_path, data_label, mean_image

def get_batch_data(data, batch_idx, batch_size, image=True):
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


class recognition_training():
    def __init__(self, epochs, batch_num, batch_size, condition=False):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.opti = tf.keras.optimizers.Adam(1e-3)
        self.condition = condition

        #set the model
        if self.condition:
            self.H_cls = H_cls()
        else:
            self.H_cls = H_cls_no_condition()

        self.encoder = encoder()
        self.ztozd = ZtoZd()
        self.ZtoZg = ZtoZg()
        self.generator = generator()
        # self.encoder.load_weights('model_weight/encoder_init')
        # self.ZtoZg.load_weights('model_weight/ztozg_init')
        self.encoder.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation')
        self.ztozd.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
        self.ZtoZg.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
        self.generator.load_weights('/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation')

        #set the data
        self.ar_train_path, self.ar_label, self.mean_ar = prepare_data()
        print(self.ar_train_path.shape, self.ar_label.shape, self.mean_ar.shape)


    def find_neighbor(self, batch_images):
        inputs = []
        for image in batch_images:
            res_image = tf.tile(tf.reshape(image, [1, 64, 64]), [self.mean_ar.shape[0], 1, 1])
            distance_mse = list(tf.reduce_mean(tf.square(self.mean_ar - res_image), axis=(1, 2)))
            # distance_mse[distance_mse.index(min(distance_mse))] = 1000

            neighbors, weights = [], []
            for i in range(2):
                # print(distance_mse.index(min(distance_mse)))
                neighbors.append(self.mean_ar[distance_mse.index(min(distance_mse))])
                weights.append(min(distance_mse))
                distance_mse[distance_mse.index(min(distance_mse))] = 1000
            neighbors, weights = np.array(neighbors), np.array(weights)
            for i in range(2):
                neighbors[i] = ((1/weights[i]) / ((1/weights[0])+(1/weights[1]))) * neighbors[i]

            neighbor_image = tf.reshape(tf.concat([tf.reshape(image, [1, 64, 64]), neighbors], axis=0), [3, 64, 64, 1])
            z, _ = self.encoder(neighbor_image)
            features_zd, _ = self.ztozd(z)
            features_zg, _, _ = self.ZtoZg(z)
            inputs.append(tf.reshape(features_zg, [-1, 200]))
        inputs = np.array(inputs)
        return inputs

    def train_step(self, real_image, label):
        label = tf.one_hot(label, 111)
        with tf.GradientTape() as tape:
            if self.condition:
                # print(np.argmax(label, axis=-1))
                real_image = self.find_neighbor(real_image)
                # print('----')
            _, pred = self.H_cls(real_image)

            cce = tf.keras.losses.CategoricalCrossentropy()
            ce_loss = cce(label, pred)
            acc = accuracy_score(np.argmax(pred, axis=-1), np.argmax(label, axis=-1))
            total_loss = ce_loss

        grads = tape.gradient(total_loss, self.H_cls.trainable_variables)
        self.opti.apply_gradients(zip(grads, self.H_cls.trainable_variables))
        return ce_loss, acc

    def training(self):
        ce_loss_epoch = []
        acc_epoch = []

        for epoch in range(1, self.epochs + 1):
            ce_loss_batch = []
            acc_batch = []

            start = time.time()
            for step in range(self.batch_num):
                print(step, end=' ')
                real_image = get_batch_data(self.ar_train_path, step, self.batch_size)
                ar_label = get_batch_data(self.ar_label, step, self.batch_size, image=False)
                ce_loss, acc = self.train_step(real_image, ar_label)

                ce_loss_batch.append(ce_loss)
                acc_batch.append(acc)
            ce_loss_epoch.append(np.mean(ce_loss_batch))
            acc_epoch.append(np.mean(acc_batch))
            print('Start of epoch %d' % (epoch))
            print(f'the ce_los is {ce_loss_epoch[-1]}')
            print(f'the accuracy is {acc_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('-----------------------------------------------')
            self.H_cls.save_weights('model_weight2/H.cls_condition')

    def rank_10(self, path='cls_datasets/train_data_var_large/', training=False):
        if self.condition:
            cls = H_cls()
            cls.load_weights('model_weight2/H.cls_condition')
        if not self.condition:
            cls = H_cls_no_condition()
            cls.load_weights('model_weight/H.cls_init')

        rank_32, rank_20, rank_16, rank_12, rank_8 = [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)]
        rank_32_syn, rank_20_syn, rank_16_syn, rank_12_syn, rank_8_syn = [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)]
        if training:
            ID = [f'ID{i}' for i in range(1, 91)]
        else:
            ID = [f'ID{i}' for i in range(91, 112)]

        number = 0
        for id in ID:
            for filename in os.listdir(path + id):
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                if '-3-0' in filename:
                    number += 1
                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(image, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low4_image = cv2.resize(cv2.resize(image, (12, 12), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low5_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    z32, _ = self.encoder(tf.reshape(low1_image, [1, 64, 64, 1]))
                    z20, _ = self.encoder(tf.reshape(low2_image, [1, 64, 64, 1]))
                    z16, _ = self.encoder(tf.reshape(low3_image, [1, 64, 64, 1]))
                    z12, _ = self.encoder(tf.reshape(low4_image, [1, 64, 64, 1]))
                    z8, _ = self.encoder(tf.reshape(low5_image, [1, 64, 64, 1]))

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

                    if self.condition:
                        feature_32g = self.find_neighbor(tf.reshape(low1_image, [1, 64, 64, 1]))
                        feature_20g = self.find_neighbor(tf.reshape(low2_image, [1, 64, 64, 1]))
                        feature_16g = self.find_neighbor(tf.reshape(low3_image, [1, 64, 64, 1]))
                        feature_12g = self.find_neighbor(tf.reshape(low4_image, [1, 64, 64, 1]))
                        feature_8g = self.find_neighbor(tf.reshape(low5_image, [1, 64, 64, 1]))


                    _, pred_32g = cls(feature_32g)
                    _, pred_20g = cls(feature_20g)
                    _, pred_16g = cls(feature_16g)
                    _, pred_12g = cls(feature_12g)
                    _, pred_8g = cls(feature_8g)

                    res_pred_32g, res_pred_20g, res_pred_16g, res_pred_12g, res_pred_8g = pred_32g.numpy(), pred_20g.numpy(), pred_16g.numpy(), pred_12g.numpy(), pred_8g.numpy()
                    res_pred = [res_pred_32g, res_pred_20g, res_pred_16g, res_pred_12g, res_pred_8g]

                    for reso, pred in enumerate(res_pred):
                        for i in range(10):
                            if np.argmax(pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:]) - 1, 111), axis=-1):
                                for j in range(i, 10):
                                    if reso == 0:
                                        rank_32[j] += 1
                                    if reso == 1:
                                        rank_20[j] += 1
                                    if reso == 2:
                                        rank_16[j] += 1
                                    if reso == 3:
                                        rank_12[j] += 1
                                    if reso == 4:
                                        rank_8[j] += 1
                                break
                            else:
                                pred[0][np.argmax(pred)] = 0

                else:
                    z, _ = self.encoder(image.reshape(1, 64, 64, 1))
                    feature_zd, _ = self.ztozd(z)
                    feature_zg, _, _ = self.ZtoZg(z)
                    if self.condition:
                        feature_zg = self.find_neighbor(tf.reshape(image, [1, 64, 64, 1]))
                        # print(f"{filename} ****")
                    _, pred = cls(feature_zg)
                    # print(f'pred{np.argmax(pred, axis=-1)}')

                    res_pred = pred.numpy()
                    for i in range(10):
                        if np.argmax(res_pred, axis=-1) == np.argmax(tf.one_hot(int(id[2:]) - 1, 111), axis=-1):
                            for j in range(i, 10):
                                if '32' in filename:
                                    rank_32_syn[j] += 1
                                if '20' in filename:
                                    rank_20_syn[j] += 1
                                if '16' in filename:
                                    rank_16_syn[j] += 1
                                if '12' in filename:
                                    rank_12_syn[j] += 1
                                if '8' in filename:
                                    rank_8_syn[j] += 1
                            break
                        else:
                            res_pred[0][np.argmax(res_pred)] = 0
        print(number)
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
        plt.plot(rank_l)
        plt.plot(rank_syn)
        plt.legend(['L', 'Syn'], loc='lower right')
        plt.title('Rank-10-Acc')
        plt.show()
        plt.close()

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


if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    recog = recognition_training(epochs=10, batch_num=111, batch_size=6, condition=True)
    recog.training()
    recog.rank_10(training=True)
    recog.rank_10(training=False)


    # data_path, data_label, mean_ar = prepare_data()
    # print(data_path[0], data_label[0])
    # image = cv2.imread(data_path[0], 0)/255
    # for i in range(1, 112):
    #     image = cv2.imread(f'cls_datasets/train_data_var_less/ID{i}/8_syn.jpg', 0)/255
    #     recog.find_neighbor(image.reshape(1, 64, 64))
    #     image = cv2.imread(f'cls_datasets/train_data_var_large/ID{i}/8_syn.jpg', 0)/255
    #     recog.find_neighbor(image.reshape(1, 64, 64))
    #     print('------')





