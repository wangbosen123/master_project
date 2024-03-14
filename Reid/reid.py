from sklearn.metrics import accuracy_score
import matplotlib.cm as cm
from regression import *


def reid_classifier():
    input1 = Input((64, 64, 2))
    input2 = Input((3))
    cond = Dense(128, activation=LeakyReLU(0.3))(input2)
    cond = Dense(512, activation=LeakyReLU(0.3))(cond)
    cond = Dense(4096, activation=LeakyReLU(0.3))(cond)
    cond = Reshape((64, 64, 1), name="reshape")(cond)
    input = tf.concat([input1, cond], axis=-1)
    x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(111, activation="softmax")(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
    model.summary()
    return model

class reid_image_classifier():
    def __init__(self, epochs, batch_num, batch_size):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.opti = tf.keras.optimizers.Adam(2e-5)
        self.cls = reid_classifier()
        self.old_cls = reid_classifier()
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('/home/bosen/gradation_thesis/0519_CGAN_synthesis_system/model_weight/AE_encoder')
        self.ztozg.load_weights('/home/bosen/gradation_thesis/0519_CGAN_synthesis_system/model_weight/zd_zg_distillation_ztozg')
        self.generator.load_weights('/home/bosen/gradation_thesis/0519_CGAN_synthesis_system/model_weight/zd_zg_distillation_generator')
        self.cls.load_weights('model_weight/cls')
        self.old_cls.load_weights('model_weight/cls')
        # for layer in self.old_cls.layers:
        #     print(layer.name)

        self.last_conv_layer_name = 'conv2d_3'
        self.network_layer_name = ['max_pooling2d_3', 'conv2d_4', 'max_pooling2d_4', 'conv2d_5', 'max_pooling2d_5', 'flatten_1',  'dense_8', 'dropout_1', 'dense_9']

    def prepare_train_data(self):
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

        train_data, test_data, train_label, test_label = [], [], [], []
        train_cond, test_cond = [], []

        for id in os.listdir(path_AR_syn_train):
            for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_train + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)

                    train_image1 = tf.concat([image, low1_image], axis=-1)
                    train_image2 = tf.concat([image, low2_image], axis=-1)
                    train_image3 = tf.concat([image, low3_image], axis=-1)
                    train_cond1 = tf.constant([1, 0, 0], dtype=tf.float32)
                    train_cond2 = tf.constant([0, 1, 0], dtype=tf.float32)
                    train_cond3 = tf.constant([0, 0, 1], dtype=tf.float32)


                    train_data.append(train_image1), train_data.append(train_image2), train_data.append(train_image3)
                    train_cond.append(train_cond1), train_cond.append(train_cond2), train_cond.append(train_cond3)
                    for i in range(3):
                        train_label.append(tf.one_hot(int(id[2:]) - 1, 111))
                else:
                    break

        for id in os.listdir(path_AR_syn_test):
            for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)


                    train_image1 = tf.concat([image, low1_image], axis=-1)
                    train_image2 = tf.concat([image, low2_image], axis=-1)
                    train_image3 = tf.concat([image, low3_image], axis=-1)
                    train_cond1 = tf.constant([1, 0, 0], dtype=tf.float32)
                    train_cond2 = tf.constant([0, 1, 0], dtype=tf.float32)
                    train_cond3 = tf.constant([0, 0, 1], dtype=tf.float32)

                    train_data.append(train_image1), train_data.append(train_image2), train_data.append(train_image3)
                    train_cond.append(train_cond1), train_cond.append(train_cond2), train_cond.append(train_cond3)
                    for i in range(3):
                        train_label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
                        # test_label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
                else:
                    break

        training_data = list(zip(train_data, train_label, train_cond))
        np.random.shuffle(training_data)
        training_data = list(zip(*training_data))
        return np.array(training_data[0]), np.array(training_data[1]), np.array(training_data[2])

    def get_batch_data(self, data, batch_idx, batch_size, image=False):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        if image:
            return np.array(train_data).reshape(-1, 64, 64, 2)
        else:
            return np.array(train_data)

    def gradcam_heatmap_mutiple(self, img_array, model, last_conv_layer_name, network_layer_name, cond, label, corresponding_label=True):
        label = np.argmax(label, axis=-1)
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

        network_input = Input(shape=last_conv_layer.output.shape[1:])
        x = network_input
        for layer_name in network_layer_name:
            x = model.get_layer(layer_name)(x)
        network_model = Model(network_input, x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model([img_array, cond])
            tape.watch(last_conv_layer_output)
            preds = network_model(last_conv_layer_output)
            if corresponding_label:
                pred_index = tf.constant(label, dtype=tf.int64)
            else:
                pred_index = np.argmax(preds, axis=-1)
            class_channel = tf.gather(preds, pred_index, axis=-1, batch_dims=1)
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        pooled_gradsa = tf.tile(tf.reshape(pooled_grads, [pooled_grads.shape[0], 1, 1, pooled_grads.shape[1]]),[1, last_conv_layer_output.shape[1], last_conv_layer_output.shape[2], 1])
        heatmap = last_conv_layer_output * pooled_gradsa
        heatmap = tf.reduce_sum(heatmap, axis=-1)
        heatmap_min, heatmap_max = [], []
        for num in range(heatmap.shape[0]):
            heatmap_min.append(np.min(heatmap[num]))
            heatmap_max.append(np.max(heatmap[num]))
        heatmap_min, heatmap_max = tf.cast(heatmap_min, dtype=tf.float32), tf.cast(heatmap_max, dtype=tf.float32)
        heatmap_min = tf.tile(tf.reshape(heatmap_min, [heatmap_min.shape[0], 1, 1]),[1, heatmap.shape[1], heatmap.shape[2]])
        heatmap_max = tf.tile(tf.reshape(heatmap_max, [heatmap_max.shape[0], 1, 1]),[1, heatmap.shape[1], heatmap.shape[2]])
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

    def train_step(self, image, cond, label, reid=False, train=True):
        with tf.GradientTape() as tape:
            att, attention_map = self.gradcam_heatmap_mutiple(image, self.old_cls, self.last_conv_layer_name, self.network_layer_name, cond, label, corresponding_label=True)
            inverse_image = tf.concat([tf.cast(tf.reshape(image[:, :, :, 0], [-1, 64, 64, 1]), dtype=tf.float32) * (1 - attention_map),
                                       tf.cast(tf.reshape(image[:, :, :, 1], [-1, 64, 64, 1]), dtype=tf.float32)], axis=-1)

            pred = self.cls([image, cond])
            pred_inv = self.cls([inverse_image, cond])
            ce_loss = self.cross_entropy_loss(label, pred)
            inv_ce_loss = self.inverse_cross_entropy_loss(label, pred_inv)

            acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred, axis=-1))

            total_loss = ce_loss
            reid_loss = inv_ce_loss + ce_loss

        if train:
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
        train_data, train_label, train_cond = self.prepare_train_data()
        print(train_data.shape, train_label.shape)

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            ce_loss_batch = []
            inv_ce_loss_batch = []
            acc_batch = []

            for batch in range(self.batch_num):
                batch_train_image = self.get_batch_data(train_data, batch, self.batch_size, image=True)
                batch_train_label = self.get_batch_data(train_label, batch, self.batch_size, image=False)
                batch_train_cond = self.get_batch_data(train_cond, batch, self.batch_size, image=False)
                ce_loss, inv_ce_loss, acc = self.train_step(batch_train_image, batch_train_cond, batch_train_label, reid=reid, train=True)

                ce_loss_batch.append(ce_loss)
                acc_batch.append(acc)
                inv_ce_loss_batch.append(inv_ce_loss)


            ce_epoch.append(np.mean(ce_loss_batch))
            acc_epoch.append(np.mean(acc_batch))
            inv_ce_epoch.append(np.mean(inv_ce_loss_batch))


            print(f'the epoch is {epoch}')
            print(f'the ce_loss is {ce_epoch[-1]}')
            print(f'the accuracy is {acc_epoch[-1]}')
            print(f'the test inv_ce_loss is {inv_ce_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')
            if reid:
                self.cls.save_weights('model_weight/reid_cls')
                self.visulaized_cam(epoch)
            else:
                self.cls.save_weights('model_weight/cls')
                self.visulaized_cam(epoch)


        if reid:
            filename = 'reid_cls'
        else:
            filename = 'cls'

        plt.plot(ce_epoch)
        plt.savefig(f'result/reid_result/{filename}_ce_loss')
        plt.close()

        plt.plot(acc_epoch)
        plt.savefig(f'result/reid_result/{filename}_acc')
        plt.close()

        plt.plot(inv_ce_epoch)
        plt.savefig(f'result/reid_result/{filename}_inv_ce_loss')
        plt.close()

    def visulaized_cam(self, epoch):
        train_data, train_label, train_cond = self.prepare_train_data()
        att, attention_map = self.gradcam_heatmap_mutiple(train_data[0: 10].reshape(10, 64, 64, 2), self.old_cls, self.last_conv_layer_name, self.network_layer_name, train_cond[0:10], train_label[0: 10], corresponding_label=True)
        plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(10):
            plt.subplot(4, 10, i + 1)
            plt.axis('off')
            plt.imshow(train_data[i][:, :, 0].reshape(64, 64), cmap='gray')

            plt.subplot(4, 10, i + 11)
            plt.axis('off')
            plt.imshow(tf.reshape(att[i], [64, 64, 3]))

            plt.subplot(4, 10, i + 21)
            plt.axis('off')
            plt.imshow(tf.reshape(attention_map[i], [64, 64]), cmap='gray')

            plt.subplot(4, 10, i + 31)
            plt.axis('off')
            inverse_image = train_data[i][:, :, 0].reshape(64, 64, 1) * (1 - attention_map[i])
            plt.imshow(tf.reshape(inverse_image, [64, 64]), cmap='gray')
        plt.savefig(f'result/reid_result/{epoch}_reid_cam')
        plt.close()


class normal_classifier():
    def __init__(self, epochs, batch_num, batch_size):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.opti = tf.keras.optimizers.Adam(5e-5)
        self.cls = reid_classifier()
        self.encoder = normal_encoder()
        self.ztozg = ZtoZg()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('/home/bosen/gradation_thesis/0519_CGAN_synthesis_system/model_weight/AE_encoder')
        self.ztozg.load_weights('/home/bosen/gradation_thesis/0519_CGAN_synthesis_system/model_weight/zd_zg_distillation_ztozg')
        self.generator.load_weights('/home/bosen/gradation_thesis/0519_CGAN_synthesis_system/model_weight/zd_zg_distillation_generator')
        self.cls.load_weights('model_weight/cls')

    def prepare_train_data(self):
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

        train_data, test_data, train_label, test_label = [], [], [], []

        for id in os.listdir(path_AR_syn_train):
            for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_train + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)

                    train_image0 = tf.concat([image, image], axis=-1)
                    train_image1 = tf.concat([low1_image, low1_image], axis=-1)
                    train_image2 = tf.concat([low2_image, low2_image], axis=-1)
                    train_image3 = tf.concat([low3_image, low3_image], axis=-1)

                    train_data.append(train_image0), train_data.append(train_image1), train_data.append(train_image2), train_data.append(train_image3)
                    for i in range(4):
                        train_label.append(tf.one_hot(int(id[2:]) - 1, 111))
                else:
                    break

        for id in os.listdir(path_AR_syn_test):
            for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)

                    train_image0 = tf.concat([image, image], axis=-1)
                    train_image1 = tf.concat([low1_image, low1_image], axis=-1)
                    train_image2 = tf.concat([low2_image, low2_image], axis=-1)
                    train_image3 = tf.concat([low3_image, low3_image], axis=-1)

                    train_data.append(train_image0), train_data.append(train_image1), train_data.append(train_image2), train_data.append(train_image3)
                    for i in range(4):
                        train_label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
                        # test_label.append(tf.one_hot(int(id[2:]) - 1 + 90, 111))
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
            return np.array(train_data).reshape(-1, 64, 64, 2)
        else:
            return np.array(train_data)

    def cross_entropy_loss(self, label, pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(label, pred)

    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            pred = self.cls(image)
            ce_loss = 5 * self.cross_entropy_loss(label, pred)
            acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred, axis=-1))
            total_loss = ce_loss
            grads = tape.gradient(total_loss, self.cls.trainable_variables)
            self.opti.apply_gradients(zip(grads, self.cls.trainable_variables))
        return ce_loss, acc

    def main(self):
        ce_epoch = []
        acc_epoch = []
        train_data, train_label = self.prepare_train_data()
        print(train_data.shape, train_label.shape)

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            ce_loss_batch = []
            acc_batch = []

            for batch in range(self.batch_num):
                batch_train_image = self.get_batch_data(train_data, batch, self.batch_size, image=True)
                batch_train_label = self.get_batch_data(train_label, batch, self.batch_size, image=False)
                ce_loss, acc = self.train_step(batch_train_image, batch_train_label)

                ce_loss_batch.append(ce_loss)
                acc_batch.append(acc)

            ce_epoch.append(np.mean(ce_loss_batch))
            acc_epoch.append(np.mean(acc_batch))

            print(f'the epoch is {epoch}')
            print(f'the ce_loss is {ce_epoch[-1]}')
            print(f'the accuracy is {acc_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')
            self.cls.save_weights('model_weight/normal_cls')

        filename = 'normal_cls'

        plt.plot(ce_epoch)
        plt.savefig(f'result/reid_result/{filename}_ce_loss')
        plt.close()

        plt.plot(acc_epoch)
        plt.savefig(f'result/reid_result/{filename}_acc')
        plt.close()



def accuracy():
    global cls
    global encoder
    global ztozg
    global generator

    cls = reid_classifier()
    encoder = normal_encoder()
    ztozg = ZtoZg()
    generator = generator()
    encoder.load_weights('/disk2/bosen/Cross-Domain-Gan-Super-Resolution/model_weight/AE_encoder')
    ztozg.load_weights("/disk2/bosen/Cross-Domain-Gan-Super-Resolution/model_weight/zd_zg_distillation_ztozg")
    generator.load_weights('/disk2/bosen/Cross-Domain-Gan-Super-Resolution/model_weight/zd_zg_distillation_generator')
    cls.load_weights('model_weight/normal_cls')
    path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
    path_AR_syn_test = "/disk2/bosen/Datasets/AR_aligment_other/"

    acc1, acc2, acc3 = 0, 0, 0
    total_number = 0
    for id in os.listdir(path_AR_syn_test):
        for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
            if count < 20:
                total_number += 1
                image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(64, 64, 1)

                z1, z2, z3 = encoder(low1_image.reshape(1, 64, 64, 1)), encoder(low2_image.reshape(1, 64, 64, 1)), encoder(low3_image.reshape(1, 64, 64, 1))
                zg1, _, _ = ztozg(z1)
                zg2, _, _ = ztozg(z2)
                zg3, _, _ = ztozg(z3)
                syn1, syn2, syn3 = generator(zg1), generator(zg2), generator(zg3)

                test_image1 = tf.reshape(tf.concat([tf.reshape(low1_image, [64, 64, 1]), low1_image], axis=-1), [1, 64, 64, 2])
                test_image2 = tf.reshape(tf.concat([tf.reshape(low2_image, [64, 64, 1]), low2_image], axis=-1), [1, 64, 64, 2])
                test_image3 = tf.reshape(tf.concat([tf.reshape(low3_image, [64, 64, 1]), low3_image], axis=-1), [1, 64, 64, 2])

                pred1, pred2, pred3 = (cls([test_image1, tf.constant([[1, 0, 0]], dtype=tf.float32)]),
                                       cls([test_image2, tf.constant([[0, 1, 0]], dtype=tf.float32)]),
                                       cls([test_image3, tf.constant([[0, 0, 1]], dtype=tf.float32)]))
                if int(id[2:]) - 1 == np.argmax(pred1, axis=-1):
                    acc1 += 1
                if int(id[2:]) - 1  == np.argmax(pred2, axis=-1):
                    acc2 += 1
                if int(id[2:]) - 1  == np.argmax(pred3, axis=-1):
                    acc3 += 1
    print(total_number)
    print(acc1 / total_number)
    print(acc2 / total_number)
    print(acc3 / total_number)

# 0.8428571428571429
# 0.8809523809523809
# 0.7619047619047619

# 1.0
# 1.0
# 0.38333333333333336



# .9952380952380953
# 0.9976190476190476
# 0.4357142857142857

# 837
# 0.4324970131421744
# 0.4133811230585424
# 0.3918757467144564

# 0.43130227001194743
# 0.4121863799283154
# 0.2054958183990442


if __name__ == '__main__':
    pass
    # reid_cls = reid_image_classifier(epochs=8, batch_num=222, batch_size=30)
    # reid_cls.main(reid=True)
    accuracy()

    # normal_cls = normal_classifier(epochs=10, batch_num=222, batch_size=30)
    # normal_cls.main()
    # accuracy()


