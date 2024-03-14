import cv2
from GAN import *
import matplotlib.pyplot as plt
import numpy as np


def pre_train_regression():
    global encoder
    global ztozd
    global ztozg
    global regression
    global generator
    encoder = normal_encoder()
    ztozg = ZtoZg()
    regression = regression()
    encoder.load_weights('model_weight/AE_encoder')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    regression.load_weights('model_weight/regression_pre_train1')

    def prepare_data():
        path_celeba_train = '/disk2/bosen/Datasets/celeba_train/'
        train_model_input, train_model_output = [], []
        test_model_input, test_model_output = [], []

        for count, filename in enumerate(os.listdir(path_celeba_train)):
            if count < 5500:
                image = cv2.imread(path_celeba_train + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                zH = encoder(image.reshape(1, 64, 64, 1))
                zh = encoder(low1_image)
                zm = encoder(low2_image)
                zl = encoder(low3_image)
                zgH, _, _ = ztozg(zH)
                zgh, _, _ = ztozg(zh)
                zgm, _, _ = ztozg(zm)
                zgl, _, _ = ztozg(zl)

                if count < 5000:
                    train_model_input.append(tf.reshape(zgH, [200]))
                    train_model_input.append(tf.reshape(zgh, [200]))
                    train_model_input.append(tf.reshape(zgm, [200]))
                    train_model_input.append(tf.reshape(zgl, [200]))
                    train_model_output.append(tf.reshape(zgH, [200]))
                    train_model_output.append(tf.reshape(zgH, [200]))
                    train_model_output.append(tf.reshape(zgH, [200]))
                    train_model_output.append(tf.reshape(zgH, [200]))

                if count >= 5000:
                    test_model_input.append(tf.reshape(zgH, [200]))
                    test_model_input.append(tf.reshape(zgh, [200]))
                    test_model_input.append(tf.reshape(zgm, [200]))
                    test_model_input.append(tf.reshape(zgl, [200]))
                    test_model_output.append(tf.reshape(zgH, [200]))
                    test_model_output.append(tf.reshape(zgH, [200]))
                    test_model_output.append(tf.reshape(zgH, [200]))
                    test_model_output.append(tf.reshape(zgH, [200]))

        return np.array(train_model_input), np.array(train_model_output), np.array(test_model_input), np.array(test_model_output)

    train_model_input, train_model_output, test_model_input, test_model_output = prepare_data()
    print(train_model_input.shape, train_model_output.shape, test_model_input.shape, test_model_output.shape)


    regression.compile(optimizer=tf.keras.optimizers.Adam(1e-6), loss='mse', metrics=['mse'])
    history = regression.fit(train_model_input, train_model_output, epochs=30, batch_size=60, validation_data=(test_model_input, test_model_output), verbose=1)
    regression.save_weights('model_weight/regression_pre_train2')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

class reg():
    def __init__(self, epochs, batch_num, batch_size):
        # set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        # set the model
        self.encoder = encoder()
        self.generator = generator()
        self.discriminator = discriminator()
        self.reg = regression()
        self.encoder.load_weights('weights/encoder')
        self.generator.load_weights('weights/generator')
        self.discriminator.load_weights('weights/discriminator')
        self.reg.load_weights('weights/regression_pre_train2')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        #prepare data
        # initialization training testing data
        self.db, self.tr_z, self.tr_zH = self.prepare_training_data(data_type='0')
        _, self.val_z, self.val_zH = self.prepare_training_data(data_type='1')
        _, self.te_z, self.te_zH = self.prepare_training_data(data_type='2')
        print(self.tr_z.shape, self.tr_zH.shape, self.val_z.shape, self.val_zH.shape, self.te_z.shape, self.te_zH.shape)

    def prepare_training_data(self, data_type='0'):
        if data_type == '0':
            celeba_path = '/disk2/bosen/Datasets/celeba_train/'
            path = '/disk2/bosen/Datasets/AR_train/'
            data_num = 5
        elif data_type == '1':
            path = '/disk2/bosen/Datasets/AR_val/'
            data_num = 5
        elif data_type == '2':
            path = '/disk2/bosen/Datasets/AR_aligment_other/'
            data_num = 10

        db, zgs, zgHs = [], [], []
        for id in os.listdir(path):
            for num, filename in enumerate(os.listdir(path + id)):
                if num == data_num:
                    break
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                zH = self.encoder(image.reshape(1, 64, 64, 1))
                zh = self.encoder(low1_image)
                zm = self.encoder(low2_image)
                zl = self.encoder(low3_image)

                zgHs.append(tf.reshape(zH, [200]))
                zgHs.append(tf.reshape(zH, [200]))
                zgHs.append(tf.reshape(zH, [200]))
                zgHs.append(tf.reshape(zH, [200]))

                zgs.append(tf.reshape(zH, [200]))
                zgs.append(tf.reshape(zh, [200]))
                zgs.append(tf.reshape(zm, [200]))
                zgs.append(tf.reshape(zl, [200]))
                db.append(tf.reshape(zH, [200]))
                db.append(tf.reshape(zh, [200]))
                db.append(tf.reshape(zm, [200]))
                db.append(tf.reshape(zl, [200]))

        if data_type == '0':
            for num, filename in enumerate(os.listdir(celeba_path)):
                if num == 450:
                    break
                image = cv2.imread(celeba_path + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                zH = self.encoder(image.reshape(1, 64, 64, 1))
                zh = self.encoder(low1_image)
                zm = self.encoder(low2_image)
                zl = self.encoder(low3_image)

                zgHs.append(tf.reshape(zH, [200]))
                zgHs.append(tf.reshape(zH, [200]))
                zgHs.append(tf.reshape(zH, [200]))
                zgHs.append(tf.reshape(zH, [200]))

                zgs.append(tf.reshape(zH, [200]))
                zgs.append(tf.reshape(zh, [200]))
                zgs.append(tf.reshape(zm, [200]))
                zgs.append(tf.reshape(zl, [200]))

        zgs, zgHs = np.array(zgs), np.array(zgHs)
        data = list(zip(zgs, zgHs))
        np.random.shuffle(data)
        data = list(zip(*data))
        db, zgs, zgHs = np.array(db), np.array(data[0]), np.array(data[1])
        return db, zgs, zgHs

    def get_batch_data(self, data, batch_idx, batch_size):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]
        return np.array(train_data)

    def distillation_loss(self, database, z, zreg):
        dis_loss = 0
        for num, latent in enumerate(z):
            latent_expand = tf.tile(tf.reshape(latent, [-1, 200]), [database.shape[0], 1])
            z_neighbor_distance = tf.sqrt(tf.reduce_sum(tf.square(latent_expand - database), axis=-1))
            z_neighbor_distance = z_neighbor_distance.numpy()
            z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)

            z_dis, zreg_dis = [0 for i in range(3)], [0 for i in range(3)]
            for i in range(3):
                min_index = np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]
                z_dis[i] += tf.reduce_sum(tf.square(latent - database[min_index]))
                zreg_dis[i] += tf.reduce_sum(tf.square(zreg[num] - self.reg(tf.reshape(database[min_index], [1, 200]))))
                z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
            dis_loss += tf.reduce_sum(tf.math.abs((tf.cast(z_dis, dtype=tf.float32)/ tf.reduce_sum(z_dis))  -  (tf.cast(zreg_dis, dtype=tf.float32)/ tf.reduce_sum(zreg_dis))))
        return dis_loss / (num + 1)

    def reg_train_step(self, z, zH, train=True):
        with tf.GradientTape() as tape:
            zreg_stage1, zreg_stage2, zreg_stage3 = self.reg(z)

            L_reg_stage1 = (tf.reduce_mean(tf.square(zH - zreg_stage1)))
            L_reg_stage2 = (tf.reduce_mean(tf.square(zH - zreg_stage2)))
            L_reg_stage3 = (tf.reduce_mean(tf.square(zH - zreg_stage3)))

            L_reg = L_reg_stage1 + L_reg_stage2 + L_reg_stage3
            L_dis = self.distillation_loss(self.db, z, zreg_stage3)
            L_total = L_reg + L_dis

        if train:
            grads = tape.gradient(L_total, self.reg.trainable_variables)
            tf.optimizers.Adam(2e-5).apply_gradients(zip(grads, self.reg.trainable_variables))
        return L_reg_stage1, L_reg_stage2, L_reg_stage3, L_dis

    def main(self):
        tr_L_reg_epoch = [[] for i in range(3)]
        val_L_reg_epoch = [[] for i in range(3)]
        te_L_reg_epoch = [[] for i in range(3)]
        tr_L_dis_epoch = []
        val_L_dis_epoch = []
        te_L_dis_epoch = []
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            tr_L_reg_batch = [[] for i in range(3)]
            val_L_reg_batch = [[] for i in range(3)]
            te_L_reg_batch = [[] for i in range(3)]
            tr_L_dis_batch = []
            val_L_dis_batch = []
            te_L_dis_batch = []

            # Train.
            for batch in range(self.batch_num):
                tr_z_batch, tr_zH_batch = self.get_batch_data(self.tr_z, batch, self.batch_size), self.get_batch_data(self.tr_zH, batch, self.batch_size)
                L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss = self.reg_train_step(tr_z_batch, tr_zH_batch, train=True)
                tr_L_reg_batch[0].append(L_reg_stage1), tr_L_reg_batch[1].append(L_reg_stage2), tr_L_reg_batch[2].append(L_reg_stage3)
                tr_L_dis_batch.append(dis_loss)

            # val.
            for batch in range(int(self.val_z.shape[0] / 10)):
                val_z_batch, val_zH_batch = self.get_batch_data(self.val_z, batch, 10), self.get_batch_data(self.val_zH, batch, 10)
                L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss = self.reg_train_step(val_z_batch, val_zH_batch, train=False)
                val_L_reg_batch[0].append(L_reg_stage1), val_L_reg_batch[1].append(L_reg_stage2), val_L_reg_batch[2].append(L_reg_stage3)
                val_L_dis_batch.append(dis_loss)

            #test.
            for batch in range(int(self.te_z.shape[0] / 10)):
                te_z_batch, te_zH_batch = self.get_batch_data(self.te_z, batch, 10), self.get_batch_data(self.te_zH, batch, 10)
                L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss = self.reg_train_step(te_z_batch, te_zH_batch, train=False)
                te_L_reg_batch[0].append(L_reg_stage1), te_L_reg_batch[1].append(L_reg_stage2), te_L_reg_batch[2].append(L_reg_stage3)
                te_L_dis_batch.append(dis_loss)


            tr_L_reg_epoch[0].append(np.mean(tr_L_reg_batch[0]))
            tr_L_reg_epoch[1].append(np.mean(tr_L_reg_batch[1]))
            tr_L_reg_epoch[2].append(np.mean(tr_L_reg_batch[2]))
            val_L_reg_epoch[0].append(np.mean(val_L_reg_batch[0]))
            val_L_reg_epoch[1].append(np.mean(val_L_reg_batch[1]))
            val_L_reg_epoch[2].append(np.mean(val_L_reg_batch[2]))
            te_L_reg_epoch[0].append(np.mean(te_L_reg_batch[0]))
            te_L_reg_epoch[1].append(np.mean(te_L_reg_batch[1]))
            te_L_reg_epoch[2].append(np.mean(te_L_reg_batch[2]))
            tr_L_dis_epoch.append(np.mean(tr_L_dis_batch))
            val_L_dis_epoch.append(np.mean(val_L_dis_batch))
            te_L_dis_epoch.append(np.mean(te_L_dis_batch))


            x = [i for i in range(1, epoch + 1)]
            x2 = [0.8 + i for i in range(epoch)]
            x3 = [0.6 + i for i in range(epoch)]
            plt.xlabel('Epoch number')
            plt.ylabel('Loss value')
            plt.title('Train Reg Loss')
            plt.bar(x, tr_L_reg_epoch[0], color='b', width=0.2)
            plt.bar(x2, tr_L_reg_epoch[1], color='r', width=0.2)
            plt.bar(x3, tr_L_reg_epoch[2], color='y', width=0.2)
            plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
            plt.savefig('result/reg_dis/train_reg_stage_error')
            plt.close()

            plt.xlabel('Epoch number')
            plt.ylabel('Loss value')
            plt.title('Val Reg Loss')
            plt.bar(x, val_L_reg_epoch[0], color='b', width=0.2)
            plt.bar(x2, val_L_reg_epoch[1], color='r', width=0.2)
            plt.bar(x3, val_L_reg_epoch[2], color='y', width=0.2)
            plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
            plt.savefig('result/reg_dis/val_reg_stage_error')
            plt.close()

            plt.xlabel('Epoch number')
            plt.ylabel('Loss value')
            plt.title('Test Reg Loss')
            plt.bar(x, te_L_reg_epoch[0], color='b', width=0.2)
            plt.bar(x2, te_L_reg_epoch[1], color='r', width=0.2)
            plt.bar(x3, te_L_reg_epoch[2], color='y', width=0.2)
            plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
            plt.savefig('result/reg_dis/test_reg_stage_error')
            plt.close()
            #########################################################
            plt.plot(tr_L_dis_epoch)
            plt.title('Train distillation Loss')
            plt.savefig('result/reg_dis/train_distillation_loss')
            plt.close()

            plt.plot(val_L_dis_epoch)
            plt.title('Val distillation Loss')
            plt.savefig('result/reg_dis/val_distillation_loss')
            plt.close()

            plt.plot(te_L_dis_epoch)
            plt.title('Test distillation Loss')
            plt.savefig('result/reg_dis/test_distillation_loss')
            plt.close()


            print(f'the epoch is {epoch}')
            print(f'the train reg 1 times loss is {tr_L_reg_epoch[0][-1]}')
            print(f'the train reg 2 times loss is {tr_L_reg_epoch[1][-1]}')
            print(f'the train reg 3 times loss is {tr_L_reg_epoch[2][-1]}')
            print(f'the train distillation is {tr_L_dis_epoch[-1]}')

            print(f'the val reg 1 times loss is {val_L_reg_epoch[0][-1]}')
            print(f'the val reg 2 times loss is {val_L_reg_epoch[1][-1]}')
            print(f'the val reg 3 times loss is {val_L_reg_epoch[2][-1]}')
            print(f'the val distillation is {val_L_dis_epoch[-1]}')

            print(f'the te reg 1 times loss is {te_L_reg_epoch[0][-1]}')
            print(f'the te reg 2 times loss is {te_L_reg_epoch[1][-1]}')
            print(f'the te reg 3 times loss is {te_L_reg_epoch[2][-1]}')
            print(f'the test distillation is {te_L_dis_epoch[-1]}')

            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.reg.save_weights(f'weights/reg_dis')

            # train_loss = [[tr_L_reg_epoch, 'train Lreg']]
            # test_loss = [[te_L_reg_epoch, 'test Lreg']]
            #
            # for tr_loss in train_loss:
            #     plt.plot(tr_loss[0][0])
            #     plt.title(f'{tr_loss[1]}_H')
            #     plt.savefig(f'result/reg/reg/{tr_loss[1]}_H_regtimes{t}')
            #     plt.close()
            #
            #     plt.plot(tr_loss[0][1])
            #     plt.title(f'{tr_loss[1]}_h')
            #     plt.savefig(f'result/reg/reg/{tr_loss[1]}_h_regtimes{t}')
            #     plt.close()
            #
            #     plt.plot(tr_loss[0][2])
            #     plt.title(f'{tr_loss[1]}_m')
            #     plt.savefig(f'result/reg/reg/{tr_loss[1]}_m_regtimes{t}')
            #     plt.close()
            #
            #     plt.plot(tr_loss[0][3])
            #     plt.title(f'{tr_loss[1]}_l')
            #     plt.savefig(f'result/reg/reg/{tr_loss[1]}_l_regtimes{t}')
            #     plt.close()
            #
            # for te_loss in test_loss:
            #     plt.plot(te_loss[0][0])
            #     plt.title(f'{te_loss[1]}_H')
            #     plt.savefig(f'result/reg/reg_without_on_demend/{te_loss[1]}_H_regtimes{t}')
            #     plt.close()
            #
            #     plt.plot(te_loss[0][1])
            #     plt.title(f'{te_loss[1]}_h')
            #     plt.savefig(f'result/reg/reg_without_on_demend/{te_loss[1]}_h_regtimes{t}')
            #     plt.close()
            #
            #     plt.plot(te_loss[0][2])
            #     plt.title(f'{te_loss[1]}_m')
            #     plt.savefig(f'result/reg/reg_without_on_demend/{te_loss[1]}_m_regtimes{t}')
            #     plt.close()
            #
            #     plt.plot(te_loss[0][3])
            #     plt.title(f'{te_loss[1]}_l')
            #     plt.savefig(f'result/reg/reg_without_on_demend/{te_loss[1]}_l_regtimes{t}')
            #     plt.close()

            # self.repeat_reg(ratio=2, train=False)
            # self.repeat_reg(ratio=4, train=False)
            # self.repeat_reg(ratio=8, train=False)

    def repeat_reg(self,  ratio, train=True):
        def down_image(image, ratio):
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

        if train: path = '/disk2/bosen/Datasets/AR_train/'
        else: path = '/disk2/bosen/Datasets/AR_test/'

        database_latent, database_high_image, database_low_image = [], [], []
        for id_num, id in enumerate(os.listdir(path)):
                for file_num, filename in enumerate(os.listdir(path + id)):
                    if 0 <= file_num < 10:
                        image = cv2.imread(path + id + '/' + filename, 0) / 255
                        image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                        blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                        low_image = down_image(tf.reshape(blur_gray, [1, 64, 64, 1]), ratio)
                        z = self.encoder(tf.reshape(low_image, [1, 64, 64, 1]))
                        zg, _, _ = self.ztozg(z)
                        database_latent.append(zg)
                        database_high_image.append(tf.reshape(image, [64, 64, 1]))
                        database_low_image.append(tf.reshape(low_image, [64, 64, 1]))

        database_latent, database_high_image, database_low_image = np.array(database_latent), np.array(database_high_image), np.array(database_low_image)
        loss = [[0 for i in range(11)] for i in range(3)]

        learning_rate, lr = [], 1
        for i in range(10):
            lr /= 0.8
            learning_rate.append(lr)

        for num_latent, (zg, high_image, low_image) in enumerate(zip(database_latent, database_high_image, database_low_image)):
            syn = self.generator(zg)
            down_syn = down_image(syn, ratio)
            loss[0][0] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
            loss[1][0] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
            loss[2][0] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
            res_loss = tf.reduce_mean(tf.square(low_image - down_syn))
            z_final = zg
            for t in range(3):
                zreg = self.reg(zg)
                dzreg = zreg - zg
                for index, w in enumerate(learning_rate):
                    z_output = zg + (w * dzreg)
                    syn = self.generator(z_output)
                    down_syn = down_image(syn, ratio)
                    if tf.reduce_mean(tf.square(low_image - down_syn)) < res_loss:
                        res_loss = tf.reduce_mean(tf.square(low_image - down_syn))
                        z_final = zg + (w * dzreg)
                    loss[t][index+1] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
                zg = z_final

        loss = np.array(loss) / (num_latent+1)
        print(loss.shape)
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        plt.plot(x, loss[0], marker='o')
        plt.plot(x, loss[1], marker='o')
        plt.plot(x, loss[2], marker='o')
        plt.xlabel('W')
        plt.ylabel('PSNR value')
        plt.legend(['Search_number=1', 'Search_number=2', 'Search_number=3'], loc='upper right')
        # x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # x_labels = [0] + learning_rate
        # plt.xticks(x_ticks, x_labels)

        for y in range(3):
            for i, j in zip(x, loss[y]):
                plt.annotate(str(j)[0: 5], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')

        if train:
            plt.savefig(f'result/reg/reg_without_on_demend/repeat_reg_ratio_{ratio}_train_psnr')
            plt.close()
        else:
            plt.savefig(f'result/reg/reg_without_on_demend/repeat_reg_ratio_{ratio}_test_psnr')
            plt.close()

def repeat_reg(ratio, train=True):
    global encoder
    global reg

    encoder = encoder()
    reg = regression()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg')

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

    if train: path = '/disk2/bosen/Datasets/AR_train/'
    else: path = '/disk2/bosen/Datasets/AR_test/'

    database_latent, database_gt = [], []
    for id_num, id in enumerate(os.listdir(path)):
            for file_num, filename in enumerate(os.listdir(path + id)):
                if 0 <= file_num < 10:
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                    low_image = down_image(tf.reshape(blur_gray, [1, 64, 64, 1]), ratio)
                    zH = encoder(image.reshape(1, 64, 64, 1))
                    z = encoder(tf.reshape(low_image, [1, 64, 64, 1]))
                    database_gt.append(zH)
                    database_latent.append(z)

    database_latent, database_gt = np.array(database_latent), np.array(database_gt)
    loss = [[0 for i in range(11)] for i in range(3)]

    learning_rate, lr = [], 1
    for i in range(10):
        lr *= 0.8
        learning_rate.append(lr)
    # learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    print(learning_rate)

    for num_latent, (zgH, zg) in enumerate(zip(database_gt, database_latent)):
        loss[0][0] += tf.reduce_mean(tf.square(zgH - zg))
        loss[1][0] += tf.reduce_mean(tf.square(zgH - zg))
        loss[2][0] += tf.reduce_mean(tf.square(zgH - zg))
        res_loss = tf.reduce_mean(tf.square(zgH - zg))
        z_final = zg
        for t in range(3):
            _, _, zreg = reg(zg)
            dzreg = zreg - zg
            for index, w in enumerate(learning_rate):
                z_output = zg + (w * dzreg)
                if tf.reduce_mean(tf.square(zgH - z_output)) < res_loss:
                    res_loss = tf.reduce_mean(tf.square(zgH - z_output))
                    z_final = zg + (w * dzreg)
                loss[t][index+1] += tf.reduce_mean(tf.square(zgH - z_output))
            zg = z_final

    loss = np.array(loss) / (num_latent+1)
    print(loss.shape)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plt.plot(x, loss[0], marker='o')
    plt.plot(x, loss[1], marker='o')
    plt.plot(x, loss[2], marker='o')
    plt.xlabel('W')
    plt.ylabel('latent MSE')
    plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')
    # x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x_labels = [0] + learning_rate
    # plt.xticks(x_ticks, x_labels)

    for y in range(3):
        for i, j in zip(x, loss[y]):
            plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.show()
    # if train:
    #     plt.savefig(f'result/reg/reg/repeat_reg_{reg_times}_ratio_{ratio}_train_psnr')
    #     plt.close()
    # else:
    #     plt.savefig(f'result/reg/reg/repeat_reg_{reg_times}_ratio_{ratio}_test_psnr')
    #     plt.close()

def reg_test(id):
    global encoder
    global generator
    global reg

    encoder = encoder()
    generator = generator()
    reg = regression()

    encoder.load_weights('weights/encoder')
    generator.load_weights('weights/generator')
    reg.load_weights('weights/reg')

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

    def visualized_image(high_image_set, low_image_set, init_syn_image, search_best, id):
        plt.subplots(figsize=(3, 6))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplot(6, 3, 1)
        plt.axis('off')
        plt.imshow(high_image_set[0], cmap='gray')

        plt.subplot(6, 3, 2)
        plt.axis('off')
        plt.imshow(high_image_set[1], cmap='gray')

        plt.subplot(6, 3, 3)
        plt.axis('off')
        plt.imshow(high_image_set[2], cmap='gray')

        plt.subplot(6, 3, 4)
        plt.axis('off')
        plt.imshow(tf.reshape(low_image_set[0], [64, 64]), cmap='gray')

        plt.subplot(6, 3, 5)
        plt.axis('off')
        plt.imshow(tf.reshape(low_image_set[1], [64, 64]), cmap='gray')

        plt.subplot(6, 3, 6)
        plt.axis('off')
        plt.imshow(tf.reshape(low_image_set[2], [64, 64]), cmap='gray')

        plt.subplot(6, 3, 7)
        plt.axis('off')
        plt.imshow(init_syn_image[0], cmap='gray')

        plt.subplot(6, 3, 8)
        plt.axis('off')
        plt.imshow(init_syn_image[1], cmap='gray')

        plt.subplot(6, 3, 9)
        plt.axis('off')
        plt.imshow(init_syn_image[2], cmap='gray')

        dis_init_low1 = tf.abs(high_image_set[0] - init_syn_image[0])
        dis_init_low2 = tf.abs(high_image_set[1] - init_syn_image[1])
        dis_init_low3 = tf.abs(high_image_set[2] - init_syn_image[2])
        dis_search_low1 = tf.abs(high_image_set[0] - tf.reshape(search_best[0][2], [64, 64]))
        dis_search_low2 = tf.abs(high_image_set[1] - tf.reshape(search_best[1][2], [64, 64]))
        dis_search_low3 = tf.abs(high_image_set[2] - tf.reshape(search_best[2][2], [64, 64]))
        print(dis_init_low3)
        print(dis_search_low3)
        print(np.mean(dis_init_low3))
        print(np.mean(dis_search_low3))

        mean_init_low1 = np.mean(dis_init_low1)
        mean_init_low2 = np.mean(dis_init_low2)
        mean_init_low3 = np.mean(dis_init_low3)
        std_init_low1 = np.mean(dis_init_low1)
        std_init_low2 = np.mean(dis_init_low2)
        std_init_low3 = np.mean(dis_init_low3)

        dis_init_low1 = np.where(dis_init_low1 > mean_init_low1 + std_init_low1, 1, 0)
        dis_init_low2 = np.where(dis_init_low2 > mean_init_low2 + std_init_low2, 1, 0)
        dis_init_low3 = np.where(dis_init_low3 > mean_init_low3 + std_init_low3, 1, 0)
        dis_search_low1 = np.where(dis_search_low1 > mean_init_low1 + std_init_low1, 1, 0)
        dis_search_low2 = np.where(dis_search_low2 > mean_init_low2 + std_init_low2, 1, 0)
        dis_search_low3 = np.where(dis_search_low3 > mean_init_low3 + std_init_low3, 1, 0)
        print(np.mean(dis_init_low3))
        print(np.mean(dis_search_low3))

        plt.subplot(6, 3, 10)
        plt.axis('off')
        plt.imshow(dis_init_low1, cmap='gray')

        plt.subplot(6, 3, 11)
        plt.axis('off')
        plt.imshow(dis_init_low2, cmap='gray')

        plt.subplot(6, 3, 12)
        plt.axis('off')
        plt.imshow(dis_init_low3, cmap='gray')

        plt.subplot(6, 3, 13)
        plt.axis('off')
        plt.imshow(tf.reshape(search_best[0][2], [64, 64]), cmap='gray')

        plt.subplot(6, 3, 14)
        plt.axis('off')
        plt.imshow(tf.reshape(search_best[1][2], [64, 64]), cmap='gray')

        plt.subplot(6, 3, 15)
        plt.axis('off')
        plt.imshow(tf.reshape(search_best[2][2], [64, 64]), cmap='gray')

        plt.subplot(6, 3, 16)
        plt.axis('off')
        plt.imshow(dis_search_low1, cmap='gray')

        plt.subplot(6, 3, 17)
        plt.axis('off')
        plt.imshow(dis_search_low2, cmap='gray')

        plt.subplot(6, 3, 18)
        plt.axis('off')
        plt.imshow(dis_search_low3, cmap='gray')

        plt.savefig(F'result/reg/reg_test/visualized_ID{id}')
        plt.close()

    # def visualized_image(high_image_set, low_image_set, init_syn_image, search_best, id):
    #     plt.subplots(figsize=(3, 9))
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.subplot(9, 3, 1)
    #     plt.axis('off')
    #     plt.imshow(high_image_set[0], cmap='gray')
    #
    #     plt.subplot(9, 3, 2)
    #     plt.axis('off')
    #     plt.imshow(high_image_set[1], cmap='gray')
    #
    #     plt.subplot(9, 3, 3)
    #     plt.axis('off')
    #     plt.imshow(high_image_set[2], cmap='gray')
    #
    #     plt.subplot(9, 3, 4)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(low_image_set[0], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 5)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(low_image_set[1], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 6)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(low_image_set[2], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 7)
    #     plt.axis('off')
    #     plt.imshow(init_syn_image[0], cmap='gray')
    #
    #     plt.subplot(9, 3, 8)
    #     plt.axis('off')
    #     plt.imshow(init_syn_image[1], cmap='gray')
    #
    #     plt.subplot(9, 3, 9)
    #     plt.axis('off')
    #     plt.imshow(init_syn_image[2], cmap='gray')
    #
    #     # dis_init_low1 = tf.abs(high_image_set[0] - init_syn_image[0])
    #     # dis_init_low2 = tf.abs(high_image_set[1] - init_syn_image[1])
    #     # dis_init_low3 = tf.abs(high_image_set[2] - init_syn_image[2])
    #
    #     dis_search1_low1 = tf.abs(init_syn_image[0] - tf.reshape(search_best[0][0], [64, 64]))
    #     dis_search1_low2 = tf.abs(init_syn_image[1] - tf.reshape(search_best[1][0], [64, 64]))
    #     dis_search1_low3 = tf.abs(init_syn_image[2] - tf.reshape(search_best[2][0], [64, 64]))
    #
    #     dis_search2_low1 = tf.abs(tf.reshape(search_best[0][0], [64, 64]) - tf.reshape(search_best[0][1], [64, 64]))
    #     dis_search2_low2 = tf.abs(tf.reshape(search_best[1][0], [64, 64]) - tf.reshape(search_best[1][1], [64, 64]))
    #     dis_search2_low3 = tf.abs(tf.reshape(search_best[2][0], [64, 64]) - tf.reshape(search_best[2][1], [64, 64]))
    #
    #     dis_search3_low1 = tf.abs(tf.reshape(search_best[0][1], [64, 64]) - tf.reshape(search_best[0][2], [64, 64]))
    #     dis_search3_low2 = tf.abs(tf.reshape(search_best[1][1], [64, 64]) - tf.reshape(search_best[1][2], [64, 64]))
    #     dis_search3_low3 = tf.abs(tf.reshape(search_best[2][1], [64, 64]) - tf.reshape(search_best[2][2], [64, 64]))
    #
    #
    #
    #     plt.subplot(9, 3, 10)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[0][0], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 11)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[1][0], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 12)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[2][0], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 13)
    #     plt.axis('off')
    #     plt.imshow(dis_search1_low1, cmap='gray')
    #
    #     plt.subplot(9, 3, 14)
    #     plt.axis('off')
    #     plt.imshow(dis_search1_low2, cmap='gray')
    #
    #     plt.subplot(9, 3, 15)
    #     plt.axis('off')
    #     plt.imshow(dis_search1_low3, cmap='gray')
    #
    #     plt.subplot(9, 3, 16)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[0][1], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 17)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[1][1], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 18)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[2][1], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 19)
    #     plt.axis('off')
    #     plt.imshow(dis_search2_low1, cmap='gray')
    #
    #     plt.subplot(9, 3, 20)
    #     plt.axis('off')
    #     plt.imshow(dis_search2_low2, cmap='gray')
    #
    #     plt.subplot(9, 3, 21)
    #     plt.axis('off')
    #     plt.imshow(dis_search2_low3, cmap='gray')
    #
    #     plt.subplot(9, 3, 22)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[0][2], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 23)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[1][2], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 24)
    #     plt.axis('off')
    #     plt.imshow(tf.reshape(search_best[2][2], [64, 64]), cmap='gray')
    #
    #     plt.subplot(9, 3, 25)
    #     plt.axis('off')
    #     plt.imshow(dis_search3_low1, cmap='gray')
    #
    #     plt.subplot(9, 3, 26)
    #     plt.axis('off')
    #     plt.imshow(dis_search3_low2, cmap='gray')
    #
    #     plt.subplot(9, 3, 27)
    #     plt.axis('off')
    #     plt.imshow(dis_search3_low3, cmap='gray')
    #
    #     plt.savefig(F'result/reg/reg_test/visualized2_ID{id}')
    #     plt.close()

    def plot_data(data, id, data_name):
        for sample in range(3):
            x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.plot(x, data[sample][0], marker='o')
            plt.plot(x, data[sample][1], marker='o')
            plt.plot(x, data[sample][2], marker='o')
            plt.xlabel('W')
            plt.ylabel(f'{data_name} value')
            plt.legend(['Search_number=1', 'Search_number=2', 'Search_number=3'], loc='upper right')

            for y in range(3):
                for i, j in zip(x, data[sample][y]):
                    plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
            plt.savefig(f'result/reg/reg_test/ID{id}_{data_name}_{2**(sample+1)}_ratio')
            plt.close()

    if id < 10:  id = f'ID0{id}'
    else:  id = f'ID{id}'

    high_image_set, low_image_set, latent_set, latent_gt_set = [], [], [], []
    test_image_path = f'/disk2/bosen/Datasets/AR_test/{id}/21_test.jpg'
    high_image = cv2.imread(test_image_path, 0) / 255
    high_image = cv2.resize(high_image, (64, 64), cv2.INTER_CUBIC)
    blur_image = cv2.GaussianBlur(high_image, (7, 7), 0)
    low1_image = down_image(tf.reshape(blur_image, [1, 64, 64, 1]), 2)
    low2_image = down_image(tf.reshape(blur_image, [1, 64, 64, 1]), 4)
    low3_image = down_image(tf.reshape(blur_image, [1 ,64, 64, 1]),8)
    z0, z1, z2, z3 = encoder(tf.reshape(high_image, [1, 64, 64, 1])), encoder(tf.reshape(low1_image, [1, 64, 64, 1])), encoder(tf.reshape(low2_image, [1, 64, 64, 1])), encoder(tf.reshape(low3_image, [1, 64, 64, 1]))

    high_image_set.append(high_image), high_image_set.append(high_image), high_image_set.append(high_image)
    low_image_set.append(low1_image), low_image_set.append(low2_image), low_image_set.append(low3_image)
    latent_gt_set.append(z0), latent_gt_set.append(z0), latent_gt_set.append(z0)
    latent_set.append(z1), latent_set.append(z2), latent_set.append(z3)
    high_image_set, low_image_set, latent_gt_set, latent_set = np.array(high_image_set), np.array(low_image_set), np.array(latent_gt_set), np.array(latent_set)

    PSNR = [[[0 for w in range(11)] for t in range(3)] for sample in range(3)]
    SSIM = [[[0 for w in range(11)] for t in range(3)] for sample in range(3)]
    score = [[[0 for w in range(11)] for t in range(3)] for sample in range(3)]
    z_reg_final = []

    learning_rate, lr = [], 1
    for i in range(10):
        lr *= 0.8
        learning_rate.append(lr)

    init_syn_image, search_best = [], [[[] for t in range(3)] for i in range(3)]
    for num_sample, (high_image, low_image, zgH, zg) in enumerate(zip(high_image_set, low_image_set, latent_gt_set, latent_set)):
        syn = generator(zg)
        init_syn_image.append(tf.reshape(syn, [64, 64]))
        #num_sample = 0(2-ratio), 1(4-ratio), 2(8-ratio).
        PSNR[num_sample][0][0] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
        PSNR[num_sample][1][0] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
        PSNR[num_sample][2][0] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
        SSIM[num_sample][0][0] += tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
        SSIM[num_sample][1][0] += tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
        SSIM[num_sample][2][0] += tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
        score[num_sample][0][0] += tf.reduce_mean(tf.square(zgH - zg))
        score[num_sample][1][0] += tf.reduce_mean(tf.square(zgH - zg))
        score[num_sample][2][0] += tf.reduce_mean(tf.square(zgH - zg))
        res_error = tf.reduce_mean(tf.square(zgH - zg))

        z_final = zg
        for t in range(3):
            zreg = reg(zg)
            dzreg = zreg - zg
            for index, w in enumerate(learning_rate):
                z_output = zg + (w * dzreg)
                syn = generator(z_output)
                if tf.reduce_mean(tf.square(zgH - z_output)) < res_error:
                    res_error = tf.reduce_mean(tf.square(zgH - z_output))
                    z_final = zg + (w * dzreg)

                PSNR[num_sample][t][index+1] += tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
                SSIM[num_sample][t][index+1] += tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0]
                score[num_sample][t][index+1] += tf.reduce_mean(tf.square(zgH - z_output))
            zg = z_final
            z_reg_final.append(z_final)
            search_best[num_sample][t].append(tf.reshape(generator(zg), [64, 64]))
    init_syn_image = np.array(init_syn_image)
    search_best = np.array(search_best)
    PSNR, SSIM, score = np.array(PSNR), np.array(SSIM), np.array(score)

    plot_data(PSNR, id, data_name='PSNR')
    plot_data(SSIM, id, data_name='SSIM')
    plot_data(score, id,data_name='score')
    visualized_image(high_image_set, low_image_set, init_syn_image, search_best, id)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    # reg = reg(epochs=20, batch_num=60, batch_size=60)
    # reg.main()

    # g = fine_tune_G(epochs=150, batch_num=30, batch_size=60)
    # g.main()

    # repeat_reg(ratio=1, train=False)
    # repeat_reg(ratio=2, train=False)
    # repeat_reg(ratio=4, train=False)
    # repeat_reg(ratio=8, train=False)









