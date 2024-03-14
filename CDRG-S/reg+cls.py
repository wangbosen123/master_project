from regression import *
from sklearn.metrics import accuracy_score


# class reg_cls():
#     def __init__(self, epochs, batch_num, batch_size):
#         # set parameters.
#         self.epochs = epochs
#         self.batch_num = batch_num
#         self.batch_size = batch_size
#
#         # set the model.
#         self.encoder = encoder()
#         self.generator = generator()
#         self.discriminator = discriminator()
#         self.reg = regression()
#         self.cls = cls()
#         self.encoder.load_weights('weights/encoder')
#         self.generator.load_weights('weights/generator')
#         self.discriminator.load_weights('weights/discriminator')
#         self.reg.load_weights('weights/reg')
#         self.cls.load_weights('weights/pretrain_cls_zreg')
#         self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#
#         # prepare data.
#         # initialization training testing data.
#         self.db, self.tr_z, self.tr_zH, self.tr_labels = self.prepare_training_data(data_type='0')
#         _, self.val_z, self.val_zH, self.val_labels = self.prepare_training_data(data_type='1')
#         _, self.te_z, self.te_zH, self.te_labels = self.prepare_training_data(data_type='2')
#         print(self.tr_z.shape, self.tr_zH.shape, self.tr_labels.shape, self.val_z.shape, self.val_zH.shape, self.val_labels.shape, self.te_z.shape, self.te_zH.shape, self.te_labels.shape)
#
#     def prepare_training_data(self, data_type='0'):
#         if data_type == '0':
#             path = '/disk2/bosen/Datasets/AR_train/'
#             data_num = 10
#         elif data_type == '1':
#             path = '/disk2/bosen/Datasets/AR_train/'
#             data_num = 5
#         elif data_type == '2':
#             path = '/disk2/bosen/Datasets/AR_aligment_other/'
#             data_num = 20
#
#         db, zgs, zgHs = [], [], []
#         labels = []
#         for id in os.listdir(path):
#             for num, filename in enumerate(os.listdir(path + id)):
#                 if num == data_num:
#                     break
#                 image = cv2.imread(path + id + '/' + filename, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#                 low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
#                 low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
#                 low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
#                 zH = self.encoder(image.reshape(1, 64, 64, 1))
#                 zh = self.encoder(low1_image)
#                 zm = self.encoder(low2_image)
#                 zl = self.encoder(low3_image)
#
#                 zgHs.append(tf.reshape(zH, [200]))
#                 zgHs.append(tf.reshape(zH, [200]))
#                 zgHs.append(tf.reshape(zH, [200]))
#                 zgHs.append(tf.reshape(zH, [200]))
#
#                 zgs.append(tf.reshape(zH, [200]))
#                 zgs.append(tf.reshape(zh, [200]))
#                 zgs.append(tf.reshape(zm, [200]))
#                 zgs.append(tf.reshape(zl, [200]))
#
#                 labels.append(tf.one_hot(int(id[2: ]) - 1, 90))
#                 labels.append(tf.one_hot(int(id[2: ]) - 1, 90))
#                 labels.append(tf.one_hot(int(id[2: ]) - 1, 90))
#                 labels.append(tf.one_hot(int(id[2: ]) - 1, 90))
#
#                 db.append(tf.reshape(zH, [200]))
#                 db.append(tf.reshape(zh, [200]))
#                 db.append(tf.reshape(zm, [200]))
#                 db.append(tf.reshape(zl, [200]))
#
#         zgs, zgHs = np.array(zgs), np.array(zgHs)
#         data = list(zip(zgs, zgHs, labels))
#         np.random.shuffle(data)
#         data = list(zip(*data))
#         db, zgs, zgHs, labels = np.array(db), np.array(data[0]), np.array(data[1]), np.array(data[2])
#         return db, zgs, zgHs, labels
#
#     def get_batch_data(self, data, batch_idx, batch_size):
#         range_min = batch_idx * batch_size
#         range_max = (batch_idx + 1) * batch_size
#
#         if range_max > len(data):
#             range_max = len(data)
#         index = list(range(range_min, range_max))
#         train_data = [data[idx] for idx in index]
#         return np.array(train_data)
#
#     def distillation_loss(self, database, z, zreg):
#         dis_loss = 0
#         for num, latent in enumerate(z):
#             latent_expand = tf.tile(tf.reshape(latent, [-1, 200]), [database.shape[0], 1])
#             z_neighbor_distance = tf.sqrt(tf.reduce_sum(tf.square(latent_expand - database), axis=-1))
#             z_neighbor_distance = z_neighbor_distance.numpy()
#             z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
#
#             z_dis, zreg_dis = [0 for i in range(3)], [0 for i in range(3)]
#             for i in range(3):
#                 min_index = np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]
#                 z_dis[i] += tf.sqrt(tf.reduce_sum(tf.square(latent - database[min_index])) + 1e-10)
#                 zreg_dis[i] += tf.sqrt(tf.reduce_sum(tf.square(zreg[num] - self.reg(tf.reshape(database[min_index], [1, 200])))) + 1e-10)
#                 z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
#             dis_loss += tf.reduce_sum(tf.math.abs((tf.cast(z_dis, dtype=tf.float32)/ tf.reduce_sum(z_dis))  -  (tf.cast(zreg_dis, dtype=tf.float32)/ tf.reduce_sum(zreg_dis))))
#         return dis_loss / int(z.shape[0])
#
#     def reg_train_step(self, z, zH, label, train_type):
#         cce = tf.keras.losses.CategoricalCrossentropy()
#         with tf.GradientTape() as tape:
#             zreg_stage1, zreg_stage2, zreg_stage3 = self.reg(z)
#             _, pred1 = self.cls(zreg_stage3)
#             _, pred2 = self.cls(zreg_stage2)
#             _, pred3 = self.cls(zreg_stage1)
#
#             # L_reg_stage1 = (tf.reduce_mean(tf.square(zH - zreg_stage1)))
#             # L_reg_stage2 = (tf.reduce_mean(tf.square(zH - zreg_stage2)))
#             # L_reg_stage3 = (tf.reduce_mean(tf.square(zH - zreg_stage3)))
#
#             # L_reg = 0.1 * (L_reg_stage1 + L_reg_stage2 + L_reg_stage3)
#             # L_dis = self.distillation_loss(self.db, z, zreg_stage3)
#             L_ce1 = 0.1 * cce(label, pred1)
#             L_ce2 = 0.1 * cce(label, pred2)
#             L_ce3 = 0.1 * cce(label, pred3)
#             L_ce = L_ce1 + L_ce2 + L_ce3
#             acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred1, axis=-1))
#             # if train_type == 'regression_cls':
#             #     L_total = L_reg + L_dis + L_ce
#             # elif train_type == 'regression':
#             #     L_total = L_reg + L_dis
#             # elif train_type == 'cls':
#             #     L_total = L_ce
#             L_reg_stage1, L_reg_stage2, L_reg_stage3, L_dis = 0, 0, 0, 0
#             L_total = L_ce
#
#         if train_type == 'regression_cls':
#             grads = tape.gradient(L_total, self.reg.trainable_variables + self.cls.trainable_variables)
#             tf.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.reg.trainable_variables + self.cls.trainable_variables))
#         elif train_type == 'regression':
#             grads = tape.gradient(L_total, self.reg.trainable_variables)
#             tf.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.reg.trainable_variables))
#         elif train_type == 'cls':
#             grads = tape.gradient(L_total, self.cls.trainable_variables)
#             tf.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.cls.trainable_variables))
#         elif train_type == None:
#             pass
#
#         return L_reg_stage1, L_reg_stage2, L_reg_stage3, L_dis, L_ce, acc
#
#     def main(self, plot=True):
#         tr_L_reg_epoch = [[] for i in range(3)]
#         val_L_reg_epoch = [[] for i in range(3)]
#         te_L_reg_epoch = [[] for i in range(3)]
#         tr_L_dis_epoch = []
#         val_L_dis_epoch = []
#         te_L_dis_epoch = []
#         tr_L_ce_epoch = []
#         val_L_ce_epoch = []
#         te_L_ce_epoch = []
#         tr_acc_epoch = []
#         val_acc_epoch = []
#         te_acc_epoch = []
#
#         for epoch in range(1, self.epochs + 1):
#             start = time.time()
#             tr_L_reg_batch = [[] for i in range(3)]
#             val_L_reg_batch = [[] for i in range(3)]
#             te_L_reg_batch = [[] for i in range(3)]
#             tr_L_dis_batch = []
#             val_L_dis_batch = []
#             te_L_dis_batch = []
#             tr_L_ce_batch = []
#             val_L_ce_batch = []
#             te_L_ce_batch = []
#             tr_acc_batch = []
#             val_acc_batch = []
#             te_acc_batch = []
#
#             if epoch <= 15: train_item = 'regression_cls'
#             elif 15 < epoch <= 30: train_item = 'regression_cls'
#             elif 30 < epoch <= 50: train_item = 'regression_cls'
#
#             # Train.
#             for batch in range(self.batch_num):
#                 tr_z_batch, tr_zH_batch, tr_label_batch = self.get_batch_data(self.tr_z, batch, self.batch_size), self.get_batch_data(self.tr_zH, batch, self.batch_size), self.get_batch_data(self.tr_labels, batch, self.batch_size)
#                 L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss, ce_loss, acc = self.reg_train_step(tr_z_batch, tr_zH_batch, tr_label_batch, train_type=train_item)
#                 tr_L_reg_batch[0].append(L_reg_stage1), tr_L_reg_batch[1].append(L_reg_stage2), tr_L_reg_batch[2].append(L_reg_stage3)
#                 tr_L_dis_batch.append(dis_loss)
#                 tr_L_ce_batch.append(ce_loss)
#                 tr_acc_batch.append(acc)
#
#             # val.
#             for batch in range(int(self.val_z.shape[0] / 10)):
#                 val_z_batch, val_zH_batch, val_label_batch = self.get_batch_data(self.val_z, batch, 10), self.get_batch_data(self.val_zH, batch, 10), self.get_batch_data(self.val_labels, batch, 10)
#                 L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss, ce_loss, acc = self.reg_train_step(val_z_batch, val_zH_batch, val_label_batch, train_type=None)
#                 val_L_reg_batch[0].append(L_reg_stage1), val_L_reg_batch[1].append(L_reg_stage2), val_L_reg_batch[2].append(L_reg_stage3)
#                 val_L_dis_batch.append(dis_loss)
#                 val_L_ce_batch.append(ce_loss)
#                 val_acc_batch.append(acc)
#
#             #test.
#             for batch in range(int(self.te_z.shape[0] / 10)):
#                 te_z_batch, te_zH_batch, te_label_batch = self.get_batch_data(self.te_z, batch, 10), self.get_batch_data(self.te_zH, batch, 10), self.get_batch_data(self.te_labels, batch, 10)
#                 L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss, ce_loss, acc = self.reg_train_step(te_z_batch, te_zH_batch, te_label_batch, train_type=None)
#                 te_L_reg_batch[0].append(L_reg_stage1), te_L_reg_batch[1].append(L_reg_stage2), te_L_reg_batch[2].append(L_reg_stage3)
#                 te_L_dis_batch.append(dis_loss)
#                 te_L_ce_batch.append(ce_loss)
#                 te_acc_batch.append(acc)
#
#             if plot:
#                 tr_L_reg_epoch[0].append(np.mean(tr_L_reg_batch[0]))
#                 tr_L_reg_epoch[1].append(np.mean(tr_L_reg_batch[1]))
#                 tr_L_reg_epoch[2].append(np.mean(tr_L_reg_batch[2]))
#                 tr_L_dis_epoch.append(np.mean(tr_L_dis_batch))
#                 tr_L_ce_epoch.append(np.mean(tr_L_ce_batch))
#                 tr_acc_epoch.append(np.mean(tr_acc_batch))
#
#                 val_L_reg_epoch[0].append(np.mean(val_L_reg_batch[0]))
#                 val_L_reg_epoch[1].append(np.mean(val_L_reg_batch[1]))
#                 val_L_reg_epoch[2].append(np.mean(val_L_reg_batch[2]))
#                 val_L_dis_epoch.append(np.mean(val_L_dis_batch))
#                 val_L_ce_epoch.append(np.mean(val_L_ce_batch))
#                 val_acc_epoch.append(np.mean(val_acc_batch))
#
#                 te_L_reg_epoch[0].append(np.mean(te_L_reg_batch[0]))
#                 te_L_reg_epoch[1].append(np.mean(te_L_reg_batch[1]))
#                 te_L_reg_epoch[2].append(np.mean(te_L_reg_batch[2]))
#                 te_L_dis_epoch.append(np.mean(te_L_dis_batch))
#                 te_L_ce_epoch.append(np.mean(te_L_ce_batch))
#                 te_acc_epoch.append(np.mean(te_acc_batch))
#
#                 # x = [i for i in range(1, epoch + 1)]
#                 # x2 = [0.8 + i for i in range(epoch)]
#                 # x3 = [0.6 + i for i in range(epoch)]
#                 # plt.xlabel('Epoch number')
#                 # plt.ylabel('Loss value')
#                 # plt.title('Train Reg Loss')
#                 # plt.bar(x, tr_L_reg_epoch[0], color='b', width=0.2)
#                 # plt.bar(x2, tr_L_reg_epoch[1], color='r', width=0.2)
#                 # plt.bar(x3, tr_L_reg_epoch[2], color='y', width=0.2)
#                 # plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
#                 # plt.savefig('result/reg_dis/train_reg_stage_error')
#                 # plt.close()
#
#                 # plt.xlabel('Epoch number')
#                 # plt.ylabel('Loss value')
#                 # plt.title('Val Reg Loss')
#                 # plt.bar(x, val_L_reg_epoch[0], color='b', width=0.2)
#                 # plt.bar(x2, val_L_reg_epoch[1], color='r', width=0.2)
#                 # plt.bar(x3, val_L_reg_epoch[2], color='y', width=0.2)
#                 # plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
#                 # plt.savefig('result/reg_dis/val_reg_stage_error')
#                 # plt.close()
#
#                 # plt.xlabel('Epoch number')
#                 # plt.ylabel('Loss value')
#                 # plt.title('Test Reg Loss')
#                 # plt.bar(x, te_L_reg_epoch[0], color='b', width=0.2)
#                 # plt.bar(x2, te_L_reg_epoch[1], color='r', width=0.2)
#                 # plt.bar(x3, te_L_reg_epoch[2], color='y', width=0.2)
#                 # plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
#                 # plt.savefig('result/reg_dis/test_reg_stage_error')
#                 # plt.close()
#                 #########################################################
#                 # plt.plot(tr_L_dis_epoch)
#                 # plt.title('Train distillation Loss')
#                 # plt.savefig('result/reg_dis/train_distillation_loss')
#                 # plt.close()
#
#                 # plt.plot(val_L_dis_epoch)
#                 # plt.title('Val distillation Loss')
#                 # plt.savefig('result/reg_dis/val_distillation_loss')
#                 # plt.close()
#                 #
#                 # plt.plot(te_L_dis_epoch)
#                 # plt.title('Test distillation Loss')
#                 # plt.savefig('result/reg_dis/test_distillation_loss')
#                 # plt.close()
#                 #########################################################
#                 plt.plot(tr_L_ce_epoch)
#                 plt.title('Train CE Loss')
#                 plt.savefig('result/reg_dis/train_ce_loss_only')
#                 plt.close()
#
#                 plt.plot(val_L_ce_epoch)
#                 plt.title('Val CE Loss')
#                 plt.savefig('result/reg_dis/val_ce_loss_only')
#                 plt.close()
#
#                 plt.plot(te_L_ce_epoch)
#                 plt.title('Test CE Loss')
#                 plt.savefig('result/reg_dis/te_ce_loss_only')
#                 plt.close()
#                 #########################################################
#                 plt.plot(tr_acc_epoch)
#                 plt.title('Train Accurace')
#                 plt.savefig('result/reg_dis/train_acc_only')
#                 plt.close()
#
#                 plt.plot(val_acc_epoch)
#                 plt.title('Val Accurace')
#                 plt.savefig('result/reg_dis/val_acc_only')
#                 plt.close()
#
#                 plt.plot(te_acc_epoch)
#                 plt.title('Test Accurace')
#                 plt.savefig('result/reg_dis/te_acc_only')
#                 plt.close()
#
#
#                 print(f'the epoch is {epoch}')
#                 print(f'the train reg 1 times loss is {tr_L_reg_epoch[0][-1]}')
#                 print(f'the train reg 2 times loss is {tr_L_reg_epoch[1][-1]}')
#                 print(f'the train reg 3 times loss is {tr_L_reg_epoch[2][-1]}')
#                 print(f'the train distillation loss is {tr_L_dis_epoch[-1]}')
#                 print(f'the train ce loss is {tr_L_ce_epoch[-1]}')
#                 print(f'the train accuracy is {tr_acc_epoch[-1]}')
#
#                 print(f'the val reg 1 times loss is {val_L_reg_epoch[0][-1]}')
#                 print(f'the val reg 2 times loss is {val_L_reg_epoch[1][-1]}')
#                 print(f'the val reg 3 times loss is {val_L_reg_epoch[2][-1]}')
#                 print(f'the val distillation is {val_L_dis_epoch[-1]}')
#                 print(f'the val ce loss is {val_L_ce_epoch[-1]}')
#                 print(f'the val accuracy is {val_acc_epoch[-1]}')
#
#                 print(f'the te reg 1 times loss is {te_L_reg_epoch[0][-1]}')
#                 print(f'the te reg 2 times loss is {te_L_reg_epoch[1][-1]}')
#                 print(f'the te reg 3 times loss is {te_L_reg_epoch[2][-1]}')
#                 print(f'the test distillation is {te_L_dis_epoch[-1]}')
#                 print(f'the test ce loss is {te_L_ce_epoch[-1]}')
#                 print(f'the test accuracy is {te_acc_epoch[-1]}')
#
#
#                 print(f'the spend time is {time.time() - start} second')
#                 print('------------------------------------------------')
#                 # self.reg.save_weights(f'weights/reg_x_cls_REG')
#                 # self.cls.save_weights(f'weights/reg_x_cls_CLS')


class reg_cls():
    def __init__(self, epochs, batch_num, batch_size):
        # set parameters.
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        # set the model.
        self.encoder = encoder()
        self.generator = generator()
        self.discriminator = discriminator()
        self.reg = regression()
        self.cls = cls()
        self.encoder.load_weights('weights/encoder')
        self.generator.load_weights('weights/generator')
        self.discriminator.load_weights('weights/discriminator')
        # self.reg.load_weights('weights/reg')
        # self.cls.load_weights('weights/pretrain_cls_zreg')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        self.reg.load_weights(f'weights/reg_x_cls_REG')
        self.cls.load_weights(f'weights/reg_x_cls_CLS')


    def prepare_training_data(self, data_type='0'):
        if data_type == '0':
            path = '/disk2/bosen/Datasets/AR_train/'
            data_num = 10
        elif data_type == '1':
            path = '/disk2/bosen/Datasets/AR_train/'
            data_num = 5
        elif data_type == '2':
            path = '/disk2/bosen/Datasets/AR_aligment_other/'
            data_num = 3

        zs, zHs, labels = [[] for i in range(90)], [[] for i in range(90)], [[] for i in range(90)]
        labels = [[] for i in range(90)]

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

                zHs[int(id[2: ]) - 1].append(tf.reshape(zH, [200]))
                zHs[int(id[2: ]) - 1].append(tf.reshape(zH, [200]))
                zHs[int(id[2: ]) - 1].append(tf.reshape(zH, [200]))
                zHs[int(id[2: ]) - 1].append(tf.reshape(zH, [200]))

                labels[int(id[2:]) - 1].append(tf.one_hot(int(id[2:]) - 1, 90))
                labels[int(id[2:]) - 1].append(tf.one_hot(int(id[2:]) - 1, 90))
                labels[int(id[2:]) - 1].append(tf.one_hot(int(id[2:]) - 1, 90))
                labels[int(id[2:]) - 1].append(tf.one_hot(int(id[2:]) - 1, 90))

                z_data = list(zip(zH, zh, zm, zl))
                np.random.shuffle(z_data)
                z_data = list(zip(*z_data))
                z_data = np.array(z_data)

                zs[int(id[2: ]) - 1].append(tf.reshape(z_data[0], [200]))
                zs[int(id[2: ]) - 1].append(tf.reshape(z_data[1], [200]))
                zs[int(id[2: ]) - 1].append(tf.reshape(z_data[2], [200]))
                zs[int(id[2: ]) - 1].append(tf.reshape(z_data[3], [200]))

        zs, zHs, labels = np.array(zs), np.array(zHs), np.array(labels)
        return zs, zHs, labels

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

    def reg_train_step(self, z, zH, label, train_type):
        cce = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            zreg_stage1, zreg_stage2, zreg_stage3 = self.reg(z)
            _, pred = self.cls(zreg_stage3)

            L_reg_stage1 = 10 * (tf.reduce_mean(tf.square(zH - zreg_stage1)))
            L_reg_stage2 = 10 * (tf.reduce_mean(tf.square(zH - zreg_stage2)))
            L_reg_stage3 = 10 * (tf.reduce_mean(tf.square(zH - zreg_stage3)))

            L_reg = (L_reg_stage1 + L_reg_stage2 + L_reg_stage3)
            L_dis = 10 * self.distillation_loss(z, zreg_stage3)
            L_ce = 0.1 * cce(label, pred)
            acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred, axis=-1))
            if train_type == 'regression_cls':
                L_total = L_reg + L_dis + L_ce
            elif train_type == 'regression':
                L_total = L_reg + L_dis
            elif train_type == 'cls':
                L_total = L_ce

        if train_type == 'regression_cls':
            grads = tape.gradient(L_total, self.reg.trainable_variables + self.cls.trainable_variables)
            tf.optimizers.Adam(1e-5).apply_gradients(zip(grads, self.reg.trainable_variables + self.cls.trainable_variables))
        elif train_type == 'regression':
            grads = tape.gradient(L_total, self.reg.trainable_variables)
            tf.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.reg.trainable_variables))
        elif train_type == 'cls':
            grads = tape.gradient(L_total, self.cls.trainable_variables)
            tf.optimizers.Adam(2e-4).apply_gradients(zip(grads, self.cls.trainable_variables))
        elif train_type == None:
            pass

        return L_reg_stage1, L_reg_stage2, L_reg_stage3, L_dis, L_ce, acc

    def main(self, plot=True):
        tr_L_reg_epoch = [[] for i in range(3)]
        val_L_reg_epoch = [[] for i in range(3)]
        te_L_reg_epoch = [[] for i in range(3)]
        tr_L_dis_epoch = []
        val_L_dis_epoch = []
        te_L_dis_epoch = []
        tr_L_ce_epoch = []
        val_L_ce_epoch = []
        te_L_ce_epoch = []
        tr_acc_epoch = []
        val_acc_epoch = []
        te_acc_epoch = []

        tr_zs, tr_zHs, tr_labels = self.prepare_training_data(data_type='0')
        val_zs, val_zHs, val_labels = self.prepare_training_data(data_type='1')
        te_zs, te_zHs, te_labels = self.prepare_training_data(data_type='2')
        print(tr_zs.shape, tr_zHs.shape, tr_labels.shape)
        print(val_zs.shape, val_zHs.shape, val_labels.shape)
        print(te_zs.shape, te_zHs.shape, te_labels.shape)

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            tr_L_reg_batch = [[] for i in range(3)]
            val_L_reg_batch = [[] for i in range(3)]
            te_L_reg_batch = [[] for i in range(3)]
            tr_L_dis_batch = []
            val_L_dis_batch = []
            te_L_dis_batch = []
            tr_L_ce_batch = []
            val_L_ce_batch = []
            te_L_ce_batch = []
            tr_acc_batch = []
            val_acc_batch = []
            te_acc_batch = []


            if epoch <= 10: train_item = 'cls'
            elif 10 < epoch <= 20: train_item = 'regression'
            elif 20 < epoch <= 50: train_item = 'regression_cls'

            # Train.
            for batch in range(tr_zs.shape[1]):
                L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss, ce_loss, acc = self.reg_train_step(tr_zs[:, batch, :], tr_zHs[:, batch, :], tr_labels[:, batch, :], train_type=train_item)
                tr_L_reg_batch[0].append(L_reg_stage1), tr_L_reg_batch[1].append(L_reg_stage2), tr_L_reg_batch[2].append(L_reg_stage3)
                tr_L_dis_batch.append(dis_loss)
                tr_L_ce_batch.append(ce_loss)
                tr_acc_batch.append(acc)

            # val.
            for batch in range(val_zs.shape[1]):
                L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss, ce_loss, acc = self.reg_train_step(val_zs[:, batch, :], val_zHs[:, batch, :], val_labels[:, batch, :], train_type=None)
                val_L_reg_batch[0].append(L_reg_stage1), val_L_reg_batch[1].append(L_reg_stage2), val_L_reg_batch[2].append(L_reg_stage3)
                val_L_dis_batch.append(dis_loss)
                val_L_ce_batch.append(ce_loss)
                val_acc_batch.append(acc)

            #test.
            for batch in range(te_zs.shape[1]):
                L_reg_stage1, L_reg_stage2, L_reg_stage3, dis_loss, ce_loss, acc = self.reg_train_step(te_zs[:, batch, :], te_zHs[:, batch, :], te_labels[:, batch, :], train_type=None)
                te_L_reg_batch[0].append(L_reg_stage1), te_L_reg_batch[1].append(L_reg_stage2), te_L_reg_batch[2].append(L_reg_stage3)
                te_L_dis_batch.append(dis_loss)
                te_L_ce_batch.append(ce_loss)
                te_acc_batch.append(acc)

            if plot:
                tr_L_reg_epoch[0].append(np.mean(tr_L_reg_batch[0]))
                tr_L_reg_epoch[1].append(np.mean(tr_L_reg_batch[1]))
                tr_L_reg_epoch[2].append(np.mean(tr_L_reg_batch[2]))
                tr_L_dis_epoch.append(np.mean(tr_L_dis_batch))
                tr_L_ce_epoch.append(np.mean(tr_L_ce_batch))
                tr_acc_epoch.append(np.mean(tr_acc_batch))

                val_L_reg_epoch[0].append(np.mean(val_L_reg_batch[0]))
                val_L_reg_epoch[1].append(np.mean(val_L_reg_batch[1]))
                val_L_reg_epoch[2].append(np.mean(val_L_reg_batch[2]))
                val_L_dis_epoch.append(np.mean(val_L_dis_batch))
                val_L_ce_epoch.append(np.mean(val_L_ce_batch))
                val_acc_epoch.append(np.mean(val_acc_batch))

                te_L_reg_epoch[0].append(np.mean(te_L_reg_batch[0]))
                te_L_reg_epoch[1].append(np.mean(te_L_reg_batch[1]))
                te_L_reg_epoch[2].append(np.mean(te_L_reg_batch[2]))
                te_L_dis_epoch.append(np.mean(te_L_dis_batch))
                te_L_ce_epoch.append(np.mean(te_L_ce_batch))
                te_acc_epoch.append(np.mean(te_acc_batch))

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
                plt.savefig('result/reg_dis2/train_reg_stage_error')
                plt.close()

                plt.xlabel('Epoch number')
                plt.ylabel('Loss value')
                plt.title('Val Reg Loss')
                plt.bar(x, val_L_reg_epoch[0], color='b', width=0.2)
                plt.bar(x2, val_L_reg_epoch[1], color='r', width=0.2)
                plt.bar(x3, val_L_reg_epoch[2], color='y', width=0.2)
                plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
                plt.savefig('result/reg_dis2/val_reg_stage_error')
                plt.close()

                plt.xlabel('Epoch number')
                plt.ylabel('Loss value')
                plt.title('Test Reg Loss')
                plt.bar(x, te_L_reg_epoch[0], color='b', width=0.2)
                plt.bar(x2, te_L_reg_epoch[1], color='r', width=0.2)
                plt.bar(x3, te_L_reg_epoch[2], color='y', width=0.2)
                plt.legend(['reg_stage1', 'reg_stage2', 'reg_stage3'], loc='upper right')
                plt.savefig('result/reg_dis2/test_reg_stage_error')
                plt.close()
                #########################################################
                plt.plot(tr_L_dis_epoch)
                plt.title('Train distillation Loss')
                plt.savefig('result/reg_dis2/train_distillation_loss')
                plt.close()

                plt.plot(val_L_dis_epoch)
                plt.title('Val distillation Loss')
                plt.savefig('result/reg_dis2/val_distillation_loss')
                plt.close()

                plt.plot(te_L_dis_epoch)
                plt.title('Test distillation Loss')
                plt.savefig('result/reg_dis2/test_distillation_loss')
                plt.close()
                #########################################################
                plt.plot(tr_L_ce_epoch)
                plt.title('Train CE Loss')
                plt.savefig('result/reg_dis2/train_ce_loss_only')
                plt.close()

                plt.plot(val_L_ce_epoch)
                plt.title('Val CE Loss')
                plt.savefig('result/reg_dis2/val_ce_loss_only')
                plt.close()

                plt.plot(te_L_ce_epoch)
                plt.title('Test CE Loss')
                plt.savefig('result/reg_dis2/te_ce_loss_only')
                plt.close()
                #########################################################
                plt.plot(tr_acc_epoch)
                plt.title('Train Accurace')
                plt.savefig('result/reg_dis2/train_acc_only')
                plt.close()

                plt.plot(val_acc_epoch)
                plt.title('Val Accurace')
                plt.savefig('result/reg_dis2/val_acc_only')
                plt.close()

                plt.plot(te_acc_epoch)
                plt.title('Test Accurace')
                plt.savefig('result/reg_dis2/te_acc_only')
                plt.close()


                print(f'the epoch is {epoch}')
                print(f'the train reg 1 times loss is {tr_L_reg_epoch[0][-1]}')
                print(f'the train reg 2 times loss is {tr_L_reg_epoch[1][-1]}')
                print(f'the train reg 3 times loss is {tr_L_reg_epoch[2][-1]}')
                print(f'the train distillation loss is {tr_L_dis_epoch[-1]}')
                print(f'the train ce loss is {tr_L_ce_epoch[-1]}')
                print(f'the train accuracy is {tr_acc_epoch[-1]}')

                print(f'the val reg 1 times loss is {val_L_reg_epoch[0][-1]}')
                print(f'the val reg 2 times loss is {val_L_reg_epoch[1][-1]}')
                print(f'the val reg 3 times loss is {val_L_reg_epoch[2][-1]}')
                print(f'the val distillation is {val_L_dis_epoch[-1]}')
                print(f'the val ce loss is {val_L_ce_epoch[-1]}')
                print(f'the val accuracy is {val_acc_epoch[-1]}')

                print(f'the te reg 1 times loss is {te_L_reg_epoch[0][-1]}')
                print(f'the te reg 2 times loss is {te_L_reg_epoch[1][-1]}')
                print(f'the te reg 3 times loss is {te_L_reg_epoch[2][-1]}')
                print(f'the test distillation is {te_L_dis_epoch[-1]}')
                print(f'the test ce loss is {te_L_ce_epoch[-1]}')
                print(f'the test accuracy is {te_acc_epoch[-1]}')


                print(f'the spend time is {time.time() - start} second')
                self.reg.save_weights(f'weights/reg_x_cls_REG')
                self.cls.save_weights(f'weights/reg_x_cls_CLS')
                self.test2_acc()
                print('------------------------------------------------')

    def test2_acc(self):
        path = '/disk2/bosen/Datasets/AR_aligment_other/'
        label = [[] for i in range(3)]
        pred = [[] for i in range(3)]

        for id in os.listdir(path):
            for num, filename in enumerate(os.listdir(path + id)):
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                zh = self.encoder(low1_image)
                zm = self.encoder(low2_image)
                zl = self.encoder(low3_image)

                _, _, zregh = self.reg(zh)
                _, _, zregm = self.reg(zm)
                _, _, zregl = self.reg(zl)

                _, p1 = self.cls(zregh)
                _, p2 = self.cls(zregm)
                _, p3 = self.cls(zregl)
                label[0].append(int(id[2:]) - 1)
                label[1].append(int(id[2:]) - 1)
                label[2].append(int(id[2:]) - 1)
                pred[0].append(np.argmax(p1, axis=-1))
                pred[1].append(np.argmax(p2, axis=-1))
                pred[2].append(np.argmax(p3, axis=-1))

        label, pred = np.array(label), np.array(pred)

        print(f'test2 2 ratio acc is {accuracy_score(label[0], pred[0])}')
        print(f'test2 4 ratio acc is {accuracy_score(label[1], pred[1])}')
        print(f'test2 8 ratio acc is {accuracy_score(label[2], pred[2])}')




if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    reg = reg_cls(epochs=30, batch_num=60, batch_size=60)
    # reg.prepare_training_data(data_type='0')
    # reg.prepare_training_data(data_type='1')
    # reg.prepare_training_data(data_type='2')
    # reg.main(plot=True)
    reg.test2_acc()













