from experiment import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier




# class reg_test():
#     def __init__(self, testing):
#         self.encoder = encoder()
#         self.reg = regression()
#         self.cls = cls()
#         self.generator = generator()
#
#         self.encoder.load_weights('weights/encoder')
#         self.generator.load_weights('weights/generator')
#         self.reg.load_weights('weights/reg_x_cls_REG')
#         self.cls.load_weights('weights/reg_x_cls_CLS')
#         self.generator.load_weights('weights/generator2')
#
#         self.testing = testing
#
#     def down_image(self, image, ratio):
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
#     def database(self):
#         z_database, zreg_database = [], []
#         path = '/disk2/bosen/Datasets/AR_train/'
#         for id in os.listdir(path):
#             for num, filename in enumerate(os.listdir(path + id)):
#                 if num ==2:
#                     break
#                 image = cv2.imread(path + id + '/' + filename, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#                 low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 z1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
#                 z2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
#                 z3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
#                 z_database.append(tf.reshape(z1, [200]))
#                 z_database.append(tf.reshape(z2, [200]))
#                 z_database.append(tf.reshape(z3, [200]))
#                 _, _, zreg1 = self.reg(z1)
#                 _, _, zreg2 = self.reg(z2)
#                 _, _, zreg3 = self.reg(z3)
#                 zreg_database.append(tf.reshape(zreg1, [200]))
#                 zreg_database.append(tf.reshape(zreg2, [200]))
#                 zreg_database.append(tf.reshape(zreg3, [200]))
#         z_database, zreg_database = np.array(z_database), np.array(zreg_database)
#         return z_database, zreg_database
#
#     def get_test_data(self, ratio):
#         if self.testing == 'testing2': path = '/disk2/bosen/Datasets/AR_aligment_other/'
#         elif self.testing == 'testing3': path = '/disk2/bosen/Datasets/AR_test/'
#
#         test_high_image, test_low_image, test_z, test_id = [], [], [], []
#         for a, id in enumerate(os.listdir(path)):
#             if a == 1:
#                 break
#             for file_num, filename in enumerate(os.listdir(path + id)):
#                 if file_num == 1:
#                     break
#                 image = cv2.imread(path + id + '/' + filename, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#
#                 if ratio == 1:
#                     low_image = image
#                 elif ratio == 2:
#                     low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 elif ratio == 4:
#                     low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 elif ratio == 8:
#                     low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#
#                 z = self.encoder(low_image.reshape(1, 64, 64, 1))
#                 test_high_image.append(image)
#                 test_low_image.append(low_image)
#                 test_z.append(z)
#                 test_id.append(int(id[2:]) - 1)
#         test_z, test_high_image, test_low_image, test_id = np.array(test_z), np.array(test_high_image), np.array(test_low_image), np.array(test_id)
#         return test_z, test_high_image, test_low_image, test_id
#
#     def rec_loss(self, gt, syn):
#         return tf.reduce_mean(tf.square(tf.reshape(tf.cast(gt, dtype=tf.float32), [-1, 64, 64, 1]) - tf.reshape(tf.cast(syn, dtype=tf.float32), [-1, 64, 64, 1])))
#
#     def reg_loss(self, z, zreg):
#         return tf.reduce_mean(tf.square(z - zreg))
#
#     def distillation_loss(self, target_z, target_zreg, database_z, database_zreg):
#         target_z_expand = tf.tile(tf.reshape(target_z, [-1, 200]), [database_z.shape[0], 1])
#         target_zreg_expand = tf.tile(tf.reshape(target_zreg, [-1, 200]), [database_zreg.shape[0], 1])
#
#         distance_target_z_database_z = tf.reduce_sum(tf.square(database_z - target_z_expand), axis=-1)
#         distance_target_zreg_database_zreg = tf.reduce_sum(tf.square(database_zreg - target_zreg_expand), axis=-1)
#
#         sum_distance_z = tf.reduce_sum(distance_target_z_database_z)
#         sum_distance_zreg = tf.reduce_sum(distance_target_zreg_database_zreg)
#
#         dis_loss = abs((distance_target_z_database_z / sum_distance_z) - (distance_target_zreg_database_zreg / sum_distance_zreg))
#         dis_loss = tf.reduce_sum(dis_loss)
#         return dis_loss
#
#     def inversion(self, latent, latent_reg, low_image, database_z, database_zreg, ratio):
#         with tf.GradientTape(persistent=True) as code_tape:
#             code_tape.watch(latent_reg)
#             syn_reg = self.generator(latent_reg)
#             rec_loss = 10 * self.rec_loss(low_image, self.down_image(syn_reg, ratio))
#             reg_loss = 0.05 * self.reg_loss(latent, latent_reg)
#             dis_loss = self.distillation_loss(latent, latent_reg, database_z, database_zreg)
#             # dis_loss = 0
#             res_total_loss = rec_loss + reg_loss + dis_loss
#             gradient_latent = code_tape.gradient(res_total_loss, latent_reg)
#         return gradient_latent, rec_loss, reg_loss, dis_loss
#
#
#     def reg_inversion(self, ratio, plot=True):
#         database_z, database_zreg = self.database()
#         test_z, test_high_image, test_low_image, test_id = self.get_test_data(ratio=ratio)
#
#         zreg_opti = []
#         init_total_loss = [[] for i in range(test_z.shape[0])]
#         rec_loss_record, reg_loss_record, dis_loss_record = [[] for i in range(test_z.shape[0])], [[] for i in range(test_z.shape[0])], [[] for i in range(test_z.shape[0])]
#         update_count = [[0 for i in range(10)] for i in range(test_z.shape[0])]
#
#         for num, (latent, high, low, id) in enumerate(zip(test_z, test_high_image, test_low_image, test_id)):
#             for step in range(1, 11):
#                 _, _, latent_reg = self.reg(latent)
#
#                 gradient_latent, rec_loss, _, dis_loss = self.inversion(latent, latent_reg, low, database_z, database_zreg, ratio)
#                 total_loss = rec_loss + dis_loss + 0.01
#                 print(rec_loss, dis_loss)
#                 print(total_loss)
#                 init_total_loss[num].append(total_loss)
#
#                 if step > 1 and (total_loss > inversion_total_loss):
#                     print('true')
#                     latent_reg = latent
#                     total_loss = inversion_total_loss
#
#                 print(total_loss)
#                 for lr in [(10*i)*(1/step) for i in range(0, 11)]:
#                     z_search = latent_reg - (lr * gradient_latent)
#                     _, rec_loss, reg_loss, dis_loss = self.inversion(latent, z_search, low, database_z, database_zreg, ratio)
#                     print(rec_loss + reg_loss + dis_loss)
#                     if (rec_loss + reg_loss + dis_loss) < total_loss:
#                         print(lr)
#                         final_rec_loss = rec_loss
#                         final_reg_loss = reg_loss
#                         final_dis_loss = dis_loss
#                         inversion_total_loss = rec_loss + reg_loss + dis_loss
#                         total_loss = inversion_total_loss
#                         print(rec_loss, reg_loss, dis_loss)
#                         print(inversion_total_loss)
#                         print('===')
#                         latent_opti = z_search
#                         update_count[num][step-1] += 1
#                     print(rec_loss + reg_loss + dis_loss)
#                 print('---------------------')
#                 latent = latent_opti
#                 rec_loss_record[num].append(final_rec_loss)
#                 reg_loss_record[num].append(final_reg_loss)
#                 dis_loss_record[num].append(final_dis_loss)
#             zreg_opti.append(latent_opti)
#
#         rec_loss_record, reg_loss_record, dis_loss_record, zreg_opti = np.array(rec_loss_record), np.array(reg_loss_record), np.array(dis_loss_record), np.array(zreg_opti)
#         init_total_loss, update_count = np.array(init_total_loss), np.array(update_count)
#         print(rec_loss_record.shape, reg_loss_record.shape, dis_loss_record.shape, zreg_opti.shape)
#         print(init_total_loss.shape, update_count.shape)
#         rec_loss_record, reg_loss_record, dis_loss_record = tf.reduce_mean(rec_loss_record, axis=0), tf.reduce_mean(reg_loss_record, axis=0), tf.reduce_mean(dis_loss_record, axis=0)
#         print(init_total_loss, update_count)
#
#         # if plot:
#         #     plt.plot(rec_loss_record)
#         #     plt.title('Rec loss')
#         #     plt.xlabel('Iterate Times')
#         #     plt.ylabel('Mean Loss value')
#         #     plt.legend(['Rec loss'], loc='upper right')
#         #     if self.testing == 'testing2':
#         #         plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_rec_loss')
#         #         plt.close()
#         #     elif self.testing == 'testing3':
#         #         plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_rec_loss')
#         #         plt.close()
#         #
#         #     plt.plot(reg_loss_record)
#         #     plt.title('Reg loss')
#         #     plt.xlabel('Iterate Times')
#         #     plt.ylabel('Mean Loss value')
#         #     plt.legend(['Reg loss'], loc='upper right')
#         #     if self.testing == 'testing2':
#         #         plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_reg_loss')
#         #         plt.close()
#         #     elif self.testing == 'testing3':
#         #         plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_reg_loss')
#         #         plt.close()
#         #
#         #
#         #     plt.plot(dis_loss_record)
#         #     plt.title('Distillation loss')
#         #     plt.xlabel('Iterate Times')
#         #     plt.ylabel('Mean Loss value')
#         #     plt.legend(['Disillation loss'], loc='upper right')
#         #     if self.testing == 'testing2':
#         #         plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_dis_loss')
#         #         plt.close()
#         #     elif self.testing == 'testing3':
#         #         plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_dis_loss')
#         #         plt.close()
#
#         z = test_z
#         _, _, zreg = self.reg(tf.reshape(z, [-1, 200]))
#         zreg_opti = zreg_opti
#         return z, zreg, zreg_opti, test_high_image, test_low_image, test_id
#
#
#     def cls_test(self, ratio):
#         _, zreg, zreg_opti, _, _, test_id = self.reg_inversion(ratio)
#         reg_acc, reg_opti_acc = 0, 0
#         for num, (latent_reg, latent_reg_opti, id) in enumerate(zip(zreg, zreg_opti, test_id)):
#             _, pred_reg = self.cls(tf.reshape(latent_reg, [1, 200]))
#             _, pred_reg_opti = self.cls(tf.reshape(latent_reg_opti, [1, 200]))
#             if np.argmax(pred_reg, axis=-1)[0] == id:
#                 reg_acc += 1
#             if np.argmax(pred_reg_opti, axis=-1)[0] == id:
#                 reg_opti_acc += 1
#
#         reg_acc, reg_opti_acc = reg_acc/(num+1), reg_opti_acc/(num+1)
#         print(reg_acc, reg_opti_acc)
#
#     def psnr_ssim(self, ratio):
#         z, zreg, zreg_opti, high_image, low_image, _ = self.reg_inversion(ratio)
#         mPSNR, mSSIM = [[] for i in range(3)], [[] for i in range(3)]
#
#
#         plt.subplots(figsize=(5, 5))
#         plt.subplots_adjust(wspace=0, hspace=0)
#         count = 0
#         for num, (latent, latent_reg, latent_reg_opti, high, low) in enumerate(zip(z, zreg, zreg_opti, high_image, low_image)):
#             syn = self.generator(latent)
#             syn_reg = self.generator(tf.reshape(latent_reg, [-1, 200]))
#             syn_reg_opti = self.generator(tf.reshape(latent_reg_opti, [-1, 200]))
#             if num > 7 and count <= 4:
#                 count += 1
#                 plt.subplot(5, 5, count)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(high, [64, 64]), cmap='gray')
#                 plt.subplot(5, 5, count+5)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(low, [64, 64]), cmap='gray')
#                 plt.subplot(5, 5, count+10)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')
#                 plt.subplot(5, 5, count+15)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(syn_reg, [64, 64]), cmap='gray')
#                 plt.subplot(5, 5, count+20)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(syn_reg_opti, [64, 64]), cmap='gray')
#             if count == 5:
#                 count = 6
#                 if self.testing == 'testing2':
#                     plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_result')
#                     plt.close()
#                 elif self.testing == 'testing3':
#                     plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_result')
#                     plt.close()
#             mPSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
#             mPSNR[1].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
#             mPSNR[2].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])
#             mSSIM[0].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
#             mSSIM[1].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
#             mSSIM[2].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])
#
#         print(mPSNR[0])
#         print(mPSNR[1])
#         print(mPSNR[2])
#         mPSNR = tf.reduce_mean(mPSNR, axis=-1)
#         mSSIM = tf.reduce_mean(mSSIM, axis=-1)
#         print(mPSNR, mSSIM)

# class reg_test():
#     def __init__(self, testing):
#         self.encoder = encoder()
#         self.reg = regression()
#         self.cls = cls()
#         self.generator = generator()
#
#         self.encoder.load_weights('weights/encoder')
#         self.generator.load_weights('weights/generator')
#         self.reg.load_weights('weights/reg_x_cls_REG')
#         self.cls.load_weights('weights/reg_x_cls_CLS')
#         self.generator.load_weights('weights/generator2')
#
#         self.testing = testing
#
#     def down_image(self, image, ratio):
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
#     def database(self):
#         z_database, zreg_database = [], []
#         path = '/disk2/bosen/Datasets/AR_train/'
#         for id in os.listdir(path):
#             for num, filename in enumerate(os.listdir(path + id)):
#                 if num ==2:
#                     break
#                 image = cv2.imread(path + id + '/' + filename, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#                 low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 z1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
#                 z2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
#                 z3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
#                 z_database.append(tf.reshape(z1, [200]))
#                 z_database.append(tf.reshape(z2, [200]))
#                 z_database.append(tf.reshape(z3, [200]))
#                 _, _, zreg1 = self.reg(z1)
#                 _, _, zreg2 = self.reg(z2)
#                 _, _, zreg3 = self.reg(z3)
#                 zreg_database.append(tf.reshape(zreg1, [200]))
#                 zreg_database.append(tf.reshape(zreg2, [200]))
#                 zreg_database.append(tf.reshape(zreg3, [200]))
#         z_database, zreg_database = np.array(z_database), np.array(zreg_database)
#         return z_database, zreg_database
#
#     def get_test_data(self, ratio):
#         if self.testing == 'testing2': path = '/disk2/bosen/Datasets/AR_aligment_other/'
#         elif self.testing == 'testing3': path = '/disk2/bosen/Datasets/AR_test/'
#
#         test_high_image, test_low_image, test_z, test_id = [], [], [], []
#         # for a, id in enumerate(os.listdir(path)):
#         #     if a == 1:
#         #         break
#         for id in os.listdir(path):
#             for file_num, filename in enumerate(os.listdir(path + id)):
#                 if file_num == 1:
#                     break
#                 image = cv2.imread(path + id + '/' + filename, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#
#                 if ratio == 1:
#                     low_image = image
#                 elif ratio == 2:
#                     low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 elif ratio == 4:
#                     low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 elif ratio == 8:
#                     low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#
#                 z = self.encoder(low_image.reshape(1, 64, 64, 1))
#                 test_high_image.append(image)
#                 test_low_image.append(low_image)
#                 test_z.append(z)
#                 test_id.append(int(id[2:]) - 1)
#         test_z, test_high_image, test_low_image, test_id = np.array(test_z), np.array(test_high_image), np.array(test_low_image), np.array(test_id)
#         return test_z, test_high_image, test_low_image, test_id
#
#     def rec_loss(self, gt, syn):
#         return tf.reduce_mean(tf.square(tf.reshape(tf.cast(gt, dtype=tf.float32), [-1, 64, 64, 1]) - tf.reshape(tf.cast(syn, dtype=tf.float32), [-1, 64, 64, 1])))
#
#
#     def distillation_loss(self, target_z, target_zreg, database_z, database_zreg):
#         target_z_expand = tf.tile(tf.reshape(target_z, [-1, 200]), [database_z.shape[0], 1])
#         target_zreg_expand = tf.tile(tf.reshape(target_zreg, [-1, 200]), [database_zreg.shape[0], 1])
#
#         distance_target_z_database_z = tf.reduce_sum(tf.square(database_z - target_z_expand), axis=-1)
#         distance_target_zreg_database_zreg = tf.reduce_sum(tf.square(database_zreg - target_zreg_expand), axis=-1)
#
#         sum_distance_z = tf.reduce_sum(distance_target_z_database_z)
#         sum_distance_zreg = tf.reduce_sum(distance_target_zreg_database_zreg)
#
#         dis_loss = abs((distance_target_z_database_z / sum_distance_z) - (distance_target_zreg_database_zreg / sum_distance_zreg))
#         dis_loss = tf.reduce_sum(dis_loss)
#         return dis_loss
#
#     def inversion(self, latent, latent_reg, low_image, database_z, database_zreg, ratio, grad=True):
#         with tf.GradientTape(persistent=True) as code_tape:
#             code_tape.watch(latent_reg)
#             syn_reg = self.generator(latent_reg)
#             rec_loss = 25 * self.rec_loss(low_image, self.down_image(syn_reg, ratio))
#             dis_loss = self.distillation_loss(latent, latent_reg, database_z, database_zreg)
#             res_total_loss = rec_loss + dis_loss
#             if grad:
#                 gradient_latent = code_tape.gradient(res_total_loss, latent_reg)
#                 return gradient_latent, rec_loss, dis_loss
#             else:
#                 return rec_loss, dis_loss
#
#     def reg_inversion(self, ratio, plot=False):
#         database_z, database_zreg = self.database()
#         test_z, test_high_image, test_low_image, test_id = self.get_test_data(ratio=ratio)
#
#         zreg_opti = []
#         init_total_loss = [[] for i in range(test_z.shape[0])]
#         rec_loss_record, dis_loss_record = [[] for i in range(test_z.shape[0])], [[] for i in range(test_z.shape[0])]
#         update_count = [[0 for i in range(11)] for i in range(test_z.shape[0])]
#         reg_times = [[0 for i in range(11)] for i in range(test_z.shape[0])]
#
#         for num, (latent, high, low, id) in enumerate(zip(test_z, test_high_image, test_low_image, test_id)):
#             print(num)
#             latent_init = latent
#             for step in range(1, 11):
#                 _, _, latent_reg = self.reg(latent)
#
#                 rec_loss, dis_loss = self.inversion(latent, latent_reg, low, database_z, database_zreg, ratio, grad=False)
#                 total_loss = rec_loss + dis_loss
#                 init_total_loss[num].append(total_loss)
#
#                 if step == 1:
#                     rec_loss_record[num].append(rec_loss)
#                     dis_loss_record[num].append(dis_loss)
#                     reg_times[num][step - 1] += 1
#
#                 if step > 1 and (total_loss < inversion_total_loss):
#                     reg_times[num][step - 1] += 1
#
#                 if step > 1 and (total_loss >= inversion_total_loss):
#                     latent_reg = latent
#                     total_loss = inversion_total_loss
#
#                 gradient_latent, rec_loss, dis_loss = self.inversion(latent, latent_reg, low, database_z, database_zreg, ratio)
#
#                 for lr in [(0.5*i)*(1/step) for i in range(0, 11)]:
#                     z_search = latent_reg - (lr * gradient_latent)
#                     _, rec_loss, dis_loss = self.inversion(latent_init, z_search, low, database_z, database_zreg, ratio)
#                     if (rec_loss + dis_loss) < total_loss:
#                         final_rec_loss = rec_loss
#                         final_dis_loss = dis_loss
#                         inversion_total_loss = rec_loss + dis_loss
#                         total_loss = inversion_total_loss
#                         latent_opti = z_search
#                         update_count[num][step-1] += 1
#
#                 latent = latent_opti
#                 rec_loss_record[num].append(final_rec_loss)
#                 dis_loss_record[num].append(final_dis_loss)
#             zreg_opti.append(latent_opti)
#
#         rec_loss_record, dis_loss_record, zreg_opti = np.array(rec_loss_record), np.array(dis_loss_record), np.array(zreg_opti)
#         init_total_loss, update_count, reg_times = np.array(init_total_loss), np.array(update_count), np.array(reg_times)
#         print(rec_loss_record.shape, dis_loss_record.shape, zreg_opti.shape)
#         print(init_total_loss.shape, update_count.shape)
#         rec_loss_record, dis_loss_record = tf.reduce_mean(rec_loss_record, axis=0), tf.reduce_mean(dis_loss_record, axis=0)
#         init_total_loss, update_count, reg_times = tf.reduce_mean(init_total_loss, axis=0), tf.reduce_mean(update_count, axis=0), tf.reduce_mean(reg_times, axis=0)
#         print(init_total_loss, update_count)
#         print(rec_loss_record, dis_loss_record)
#
#         if plot:
#             x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#             plt.plot(x_coordinates, rec_loss_record, marker='o')
#             plt.xticks(x_coordinates)
#             plt.title('Rec loss')
#             plt.xlabel('Iterate Times')
#             plt.ylabel('Mean Loss value')
#             plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#             plt.legend(['Rec loss'], loc='upper right')
#             if self.testing == 'testing2':
#                 plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_rec_loss')
#                 plt.close()
#             elif self.testing == 'testing3':
#                 plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_rec_loss')
#                 plt.close()
#
#             x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#             plt.plot(x_coordinates, dis_loss_record, marker='o')
#             plt.xticks(x_coordinates)
#             plt.title('Distillation loss')
#             plt.xlabel('Iterate Times')
#             plt.ylabel('Mean Loss value')
#             plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#             plt.legend(['Distillation loss'], loc='upper right')
#             if self.testing == 'testing2':
#                 plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_dis_loss')
#                 plt.close()
#             elif self.testing == 'testing3':
#                 plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_dis_loss')
#                 plt.close()
#
#             x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#             plt.plot(x_coordinates, init_total_loss, marker='o', label='Record Reg Total Loss')
#             plt.xticks(x_coordinates)
#             plt.title('Record total loss')
#             plt.xlabel('Iterate Times')
#             plt.ylabel('Mean Loss value')
#             plt.legend(['Record total loss'], loc='upper right')
#             if self.testing == 'testing2':
#                 plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_Record_total_loss')
#                 plt.close()
#             elif self.testing == 'testing3':
#                 plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_Record_total_loss')
#                 plt.close()
#
#             x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#             plt.plot(x_coordinates, update_count, marker='o', label='Update Times')
#             plt.xticks(x_coordinates)
#             plt.title('update_count')
#             plt.xlabel('iterate times')
#             plt.ylabel('update times')
#             plt.legend(['update_count'], loc='upper right')
#             if self.testing == 'testing2':
#                 plt.savefig(f'result/reg_test/90id_var_{ratio}_update_count')
#                 plt.close()
#             elif self.testing == 'testing3':
#                 plt.savefig(f'result/reg_test/21id_var_{ratio}_update_count')
#                 plt.close()
#
#             x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#             plt.plot(x_coordinates, reg_times, marker='o', label='Reg Times')
#             plt.xticks(x_coordinates)
#             plt.title('reg_times')
#             plt.xlabel('iterate times')
#             plt.ylabel('reg times')
#             plt.legend(['reg_times'], loc='upper right')
#             if self.testing == 'testing2':
#                 plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_reg_times')
#                 plt.close()
#             elif self.testing == 'testing3':
#                 plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_reg_times')
#                 plt.close()
#
#         z = test_z
#         _, _, zreg = self.reg(tf.reshape(z, [-1, 200]))
#         zreg_opti = zreg_opti
#         return z, zreg, zreg_opti, test_high_image, test_low_image, test_id
#
#
#     def cls_test(self, ratio):
#         _, zreg, zreg_opti, _, _, test_id = self.reg_inversion(ratio)
#         reg_acc, reg_opti_acc = 0, 0
#         for num, (latent_reg, latent_reg_opti, id) in enumerate(zip(zreg, zreg_opti, test_id)):
#             _, pred_reg = self.cls(tf.reshape(latent_reg, [1, 200]))
#             _, pred_reg_opti = self.cls(tf.reshape(latent_reg_opti, [1, 200]))
#             if np.argmax(pred_reg, axis=-1)[0] == id:
#                 reg_acc += 1
#             if np.argmax(pred_reg_opti, axis=-1)[0] == id:
#                 reg_opti_acc += 1
#
#         reg_acc, reg_opti_acc = reg_acc/(num+1), reg_opti_acc/(num+1)
#         print(reg_acc, reg_opti_acc)
#
#     def psnr_ssim(self, ratio):
#         z, zreg, zreg_opti, high_image, low_image, _ = self.reg_inversion(ratio)
#         mPSNR, mSSIM = [[] for i in range(3)], [[] for i in range(3)]
#
#
#         plt.subplots(figsize=(7, 7))
#         plt.subplots_adjust(wspace=0, hspace=0)
#         count = 0
#         for num, (latent, latent_reg, latent_reg_opti, high, low) in enumerate(zip(z, zreg, zreg_opti, high_image, low_image)):
#             syn = self.generator(latent)
#             syn_reg = self.generator(tf.reshape(latent_reg, [-1, 200]))
#             syn_reg_opti = self.generator(tf.reshape(latent_reg_opti, [-1, 200]))
#             if num > 7 and count <= 6:
#                 count += 1
#                 plt.subplot(7, 7, count)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(high, [64, 64]), cmap='gray')
#
#                 plt.subplot(7, 7, count+7)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(low, [64, 64]), cmap='gray')
#
#                 plt.subplot(7, 7, count+14)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')
#
#                 plt.subplot(7, 7, count+21)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(syn_reg, [64, 64]), cmap='gray')
#
#                 plt.subplot(7, 7, count+28)
#                 plt.axis('off')
#                 plt.imshow(tf.reshape(syn_reg_opti, [64, 64]), cmap='gray')
#
#                 plt.subplot(7, 7, count + 35)
#                 plt.axis('off')
#                 diff_syn_reg = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg, [64, 64]).numpy())
#                 diff_syn_reg_binary = (diff_syn_reg > np.mean(diff_syn_reg) + 0.5*np.std(diff_syn_reg)).astype(int)
#                 plt.imshow(tf.reshape(diff_syn_reg_binary, [64, 64]), cmap='gray')
#
#                 plt.subplot(7, 7, count + 42)
#                 plt.axis('off')
#                 diff = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg_opti, [64, 64]).numpy())
#                 diff_binary = (diff > np.mean(diff_syn_reg) + 0.5*np.std(diff_syn_reg)).astype(int)
#                 plt.imshow(tf.reshape(diff_binary, [64, 64]), cmap='gray')
#
#             if count == 7:
#                 count = 8
#                 if self.testing == 'testing2':
#                     plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_result')
#                     plt.close()
#                 elif self.testing == 'testing3':
#                     plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_result')
#                     plt.close()
#             mPSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
#             mPSNR[1].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
#             mPSNR[2].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])
#             mSSIM[0].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
#             mSSIM[1].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
#             mSSIM[2].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])
#
#         print(mPSNR[0])
#         print(mPSNR[1])
#         print(mPSNR[2])
#         mPSNR = tf.reduce_mean(mPSNR, axis=-1)
#         mSSIM = tf.reduce_mean(mSSIM, axis=-1)
#         print(mPSNR, mSSIM)


class reg_test():
    def __init__(self, testing):
        self.encoder = encoder()
        self.reg = regression()
        self.cls = cls()
        self.generator = generator()

        self.encoder.load_weights('weights/encoder')
        self.reg.load_weights('weights/reg_x_cls_REG')
        self.cls.load_weights('weights/reg_x_cls_CLS')
        self.generator.load_weights('weights/generator2')

        self.testing = testing

    def down_image(self, image, ratio):
        if ratio == 1:
            return tf.cast(image, dtype=tf.float32)
        elif ratio == 2:
            down_syn = tf.image.resize(image, [32, 32], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 3.2:
            down_syn = tf.image.resize(image, [20, 20], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 4:
            down_syn = tf.image.resize(image, [16, 16], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 6.4:
            down_syn = tf.image.resize(image, [10, 10], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn
        elif ratio == 8:
            down_syn = tf.image.resize(image, [8, 8], method='bicubic')
            down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
            return down_syn

    def database(self):
        z_database, zreg_database = [], []
        path = '/disk2/bosen/Datasets/AR_train/'
        for id in os.listdir(path):
            for num, filename in enumerate(os.listdir(path + id)):
                if num ==2:
                    break
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                z1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                z2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                z3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
                z_database.append(tf.reshape(z1, [200]))
                z_database.append(tf.reshape(z2, [200]))
                z_database.append(tf.reshape(z3, [200]))
                _, _, zreg1 = self.reg(z1)
                _, _, zreg2 = self.reg(z2)
                _, _, zreg3 = self.reg(z3)
                zreg_database.append(tf.reshape(zreg1, [200]))
                zreg_database.append(tf.reshape(zreg2, [200]))
                zreg_database.append(tf.reshape(zreg3, [200]))
        z_database, zreg_database = np.array(z_database), np.array(zreg_database)
        return z_database, zreg_database

    def get_test_data(self, ratio, kernel):
        if self.testing == 'testing2': path = '/disk2/bosen/Datasets/Train/'
        elif self.testing == 'testing3': path = '/disk2/bosen/Datasets/AR_test/'

        test_high_image, test_low_image, test_z, test_id = [], [], [], []
        for id in os.listdir(path):
            for file_num, filename in enumerate(os.listdir(path + id)):
        # for filename in os.listdir(path):
                if file_num == 1:
                    break
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                # image = cv2.imread(path + filename, 0)/ 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, kernel, 0)

                if ratio == 1:
                    low_image = image
                elif ratio == 2:
                    low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 3.2:
                    low_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 4:
                    low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 6.4:
                    low_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 8:
                    low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                z = self.encoder(low_image.reshape(1, 64, 64, 1))
                test_high_image.append(image)
                test_low_image.append(low_image)
                test_z.append(z)
                # test_id.append(int(id[2:]))
                test_id.append(0)
        test_z, test_high_image, test_low_image, test_id = np.array(test_z), np.array(test_high_image), np.array(test_low_image), np.array(test_id)
        return test_z, test_high_image, test_low_image, test_id

    def rec_loss(self, gt, syn):
        return tf.reduce_mean(tf.square(tf.reshape(tf.cast(gt, dtype=tf.float32), [-1, 64, 64, 1]) - tf.reshape(tf.cast(syn, dtype=tf.float32), [-1, 64, 64, 1])))

    def distillation_loss(self, target_z, target_zreg, database_z, database_zreg):
        target_z_expand = tf.tile(tf.reshape(target_z, [-1, 200]), [database_z.shape[0], 1])
        target_zreg_expand = tf.tile(tf.reshape(target_zreg, [-1, 200]), [database_zreg.shape[0], 1])

        distance_target_z_database_z = tf.reduce_sum(tf.square(database_z - target_z_expand), axis=-1)
        distance_target_zreg_database_zreg = tf.reduce_sum(tf.square(database_zreg - target_zreg_expand), axis=-1)

        sum_distance_z = tf.reduce_sum(distance_target_z_database_z)
        sum_distance_zreg = tf.reduce_sum(distance_target_zreg_database_zreg)

        dis_loss = abs((distance_target_z_database_z / sum_distance_z) - (distance_target_zreg_database_zreg / sum_distance_zreg))
        dis_loss = tf.reduce_sum(dis_loss)
        return dis_loss

    def inversion(self, latent, latent_reg, low_image, database_z, database_zreg, ratio, grad=True):
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(latent_reg)
            syn_reg = self.generator(latent_reg)
            rec_loss = 25 * self.rec_loss(low_image, self.down_image(syn_reg, ratio))
            dis_loss = self.distillation_loss(latent, latent_reg, database_z, database_zreg)
            res_total_loss = rec_loss + dis_loss
            if grad:
                gradient_latent = code_tape.gradient(res_total_loss, latent_reg)
                return gradient_latent, rec_loss, dis_loss
            else:
                return rec_loss, dis_loss

    def reg_inversion(self, ratio, kernel, plot=True):
        database_z, database_zreg = self.database()
        test_z, test_high_image, test_low_image, test_id = self.get_test_data(ratio=ratio, kernel=kernel)
        print(test_z.shape, test_high_image.shape, test_low_image.shape, test_id.shape)

        zreg_opti = []
        init_total_loss = [[] for i in range(test_z.shape[0])]
        rec_loss_record, dis_loss_record = [[] for i in range(test_z.shape[0])], [[] for i in range(test_z.shape[0])]
        update_count = [[0 for i in range(11)] for i in range(test_z.shape[0])]
        reg_times = [[0 for i in range(11)] for i in range(test_z.shape[0])]

        for num, (latent, high, low, id) in enumerate(zip(test_z, test_high_image, test_low_image, test_id)):
            print(num, id)
            latent_init = latent
            for step in range(1, 11):
                _, _, latent_reg = self.reg(latent)

                rec_loss, dis_loss = self.inversion(latent_init, latent_reg, low, database_z, database_zreg, ratio, grad=False)
                total_loss = rec_loss + dis_loss
                init_total_loss[num].append(total_loss)

                if step == 1:
                    rec_loss_record[num].append(rec_loss)
                    dis_loss_record[num].append(dis_loss)
                    reg_times[num][step - 1] += 1

                # if step > 1:
                #     print(rec_loss, dis_loss, final_rec_loss, final_dis_loss)
                #
                # if step > 1 and (total_loss < inversion_total_loss):
                #     inversion_total_loss = total_loss
                #     reg_times[num][step - 1] += 1

                if step > 1 and (total_loss > inversion_total_loss):
                    latent_reg = latent
                    total_loss = inversion_total_loss

                gradient_latent, rec_loss, dis_loss = self.inversion(latent_init, latent_reg, low, database_z, database_zreg, ratio)

                for lr in [(0.3*i)*(1/step) for i in range(0, 11)]:
                    z_search = latent_reg - (lr * gradient_latent)
                    _, rec_loss, dis_loss = self.inversion(latent_init, z_search, low, database_z, database_zreg, ratio)
                    if (rec_loss + dis_loss) < total_loss:
                        final_rec_loss = rec_loss
                        final_dis_loss = dis_loss
                        inversion_total_loss = rec_loss + dis_loss
                        total_loss = inversion_total_loss
                        latent_opti = z_search
                        update_count[num][step-1] += 1

                latent = latent_opti
                rec_loss_record[num].append(final_rec_loss)
                dis_loss_record[num].append(final_dis_loss)
            zreg_opti.append(latent_opti)
            if kernel == (1, 1): np.save(f'reg_opti_data/DBt2/ratio{ratio}/kernel1/{num}.npy', latent_opti)
            if kernel == (3, 3): np.save(f'reg_opti_data/DBt2/ratio{ratio}/kernel3/{num}.npy', latent_opti)
            if kernel == (7, 7): np.save(f'reg_opti_data/DBt2/ratio{ratio}/kernel7/{num}.npy', latent_opti)
            if kernel == (11, 11): np.save(f'reg_opti_data/DBt2/ratio{ratio}/kernel11/{num}.npy', latent_opti)


        rec_loss_record, dis_loss_record, zreg_opti = np.array(rec_loss_record), np.array(dis_loss_record), np.array(zreg_opti)
        init_total_loss, update_count, reg_times = np.array(init_total_loss), np.array(update_count), np.array(reg_times)
        print(rec_loss_record.shape, dis_loss_record.shape, zreg_opti.shape)
        print(init_total_loss.shape, update_count.shape)
        rec_loss_record, dis_loss_record = tf.reduce_mean(rec_loss_record, axis=0), tf.reduce_mean(dis_loss_record, axis=0)
        init_total_loss, update_count, reg_times = tf.reduce_mean(init_total_loss, axis=0), tf.reduce_mean(update_count, axis=0), tf.reduce_mean(reg_times, axis=0)
        print(init_total_loss, update_count)
        print(rec_loss_record, dis_loss_record)

        # if plot:
        #     x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #     plt.plot(x_coordinates, rec_loss_record, marker='o')
        #     plt.xticks(x_coordinates)
        #     plt.title('Rec loss')
        #     plt.xlabel('Iterate Times')
        #     plt.ylabel('Mean Loss value')
        #     plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        #     plt.legend(['Rec loss'], loc='upper right')
        #     if self.testing == 'testing2':
        #         plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_rec_loss')
        #         plt.close()
        #     elif self.testing == 'testing3':
        #         plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_rec_loss')
        #         plt.close()
        #
        #     x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #     plt.plot(x_coordinates, dis_loss_record, marker='o')
        #     plt.xticks(x_coordinates)
        #     plt.title('Distillation loss')
        #     plt.xlabel('Iterate Times')
        #     plt.ylabel('Mean Loss value')
        #     plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        #     plt.legend(['Distillation loss'], loc='upper right')
        #     if self.testing == 'testing2':
        #         plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_dis_loss')
        #         plt.close()
        #     elif self.testing == 'testing3':
        #         plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_dis_loss')
        #         plt.close()
        #
        #     x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #     plt.plot(x_coordinates, init_total_loss, marker='o', label='Record Reg Total Loss')
        #     plt.xticks(x_coordinates)
        #     plt.title('Record total loss')
        #     plt.xlabel('Iterate Times')
        #     plt.ylabel('Mean Loss value')
        #     plt.legend(['Record total loss'], loc='upper right')
        #     if self.testing == 'testing2':
        #         plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_Record_total_loss')
        #         plt.close()
        #     elif self.testing == 'testing3':
        #         plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_Record_total_loss')
        #         plt.close()
        #
        #     x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #     plt.plot(x_coordinates, update_count, marker='o', label='Update Times')
        #     plt.xticks(x_coordinates)
        #     plt.title('update_count')
        #     plt.xlabel('iterate times')
        #     plt.ylabel('update times')
        #     plt.legend(['update_count'], loc='upper right')
        #     if self.testing == 'testing2':
        #         plt.savefig(f'result/reg_test/90id_var_{ratio}_update_count')
        #         plt.close()
        #     elif self.testing == 'testing3':
        #         plt.savefig(f'result/reg_test/21id_var_{ratio}_update_count')
        #         plt.close()
        #
        #     # x_coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #     # plt.plot(x_coordinates, reg_times, marker='o', label='Reg Times')
        #     # plt.xticks(x_coordinates)
        #     # plt.title('reg_times')
        #     # plt.xlabel('iterate times')
        #     # plt.ylabel('reg times')
        #     # plt.legend(['reg_times'], loc='upper right')
        #     # if self.testing == 'testing2':
        #     #     plt.savefig(f'result/reg_test/90id_var_{ratio}_ratio_reg_times')
        #     #     plt.close()
        #     # elif self.testing == 'testing3':
        #     #     plt.savefig(f'result/reg_test/21id_var_{ratio}_ratio_reg_times')
        #     #     plt.close()

        z = test_z
        _, _, zreg = self.reg(tf.reshape(z, [-1, 200]))
        zreg_opti = zreg_opti
        return z, zreg, zreg_opti, test_high_image, test_low_image, test_id

    def cls_test(self):
        _, zreg, zreg_opti, _, _, test_id = self.reg_inversion(ratio)
        reg_acc, reg_opti_acc = 0, 0
        for num, (latent_reg, latent_reg_opti, id) in enumerate(zip(zreg, zreg_opti, test_id)):
            _, pred_reg = self.cls(tf.reshape(latent_reg, [1, 200]))
            _, pred_reg_opti = self.cls(tf.reshape(latent_reg_opti, [1, 200]))
            if np.argmax(pred_reg, axis=-1)[0] == id:
                reg_acc += 1
            if np.argmax(pred_reg_opti, axis=-1)[0] == id:
                reg_opti_acc += 1

        reg_acc, reg_opti_acc = reg_acc/(num+1), reg_opti_acc/(num+1)
        print(reg_acc, reg_opti_acc)

    def knn_test(self):
        path_train = '/disk2/bosen/Datasets/AR_train/'
        path_test = f'90ID_other/'
        path_test = '/disk2/bosen/Datasets/AR_aligment_other/'

        database_feature, database_label = [], []
        feature2ratio, feature4ratio, feature8ratio = [], [], []
        feature2ratio_label, feature4ratio_label, feature8ratio_label = [], [], []

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

                    feature = self.encoder(image.reshape(1, 64, 64, 1))
                    feature1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    feature2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    feature3 = self.encoder(low3_image.reshape(1, 64, 64, 1))

                    # _, _, feature = self.reg(z0)
                    # _, _, feature1 = self.reg(z1)
                    # _, _, feature2 = self.reg(z2)
                    # _, _, feature3 = self.reg(z3)


                    feature = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
                    feature1 = feature1 / tf.sqrt(tf.reduce_sum(tf.square(feature1)))
                    feature2 = feature2 / tf.sqrt(tf.reduce_sum(tf.square(feature2)))
                    feature3 = feature3 / tf.sqrt(tf.reduce_sum(tf.square(feature3)))

                    database_feature.append(tf.reshape(feature, [200]))
                    database_feature.append(tf.reshape(feature1, [200]))
                    database_feature.append(tf.reshape(feature2, [200]))
                    database_feature.append(tf.reshape(feature3, [200]))

                    database_label.append(int(id[2:]))
                    database_label.append(int(id[2:]))
                    database_label.append(int(id[2:]))
                    database_label.append(int(id[2:]))



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

        for id in os.listdir(path_test):
            for num, filename in enumerate(os.listdir(path_test + id)):
                image = cv2.imread(path_test + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                feature1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                feature2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                feature3 = self.encoder(low3_image.reshape(1, 64, 64, 1))

                # _, _, feature1 = self.reg(z1)
                # _, _, feature2 = self.reg(z2)
                # _, _, feature3 = self.reg(z3)

                feature1 = feature1 / tf.sqrt(tf.reduce_sum(tf.square(feature1)))
                feature2 = feature2 / tf.sqrt(tf.reduce_sum(tf.square(feature2)))
                feature3 = feature3 / tf.sqrt(tf.reduce_sum(tf.square(feature3)))

                feature2ratio.append(tf.reshape(feature1, [200]))
                feature2ratio_label.append(int(id[2:]))
                feature4ratio.append(tf.reshape(feature2, [200]))
                feature4ratio_label.append(int(id[2:]))
                feature8ratio.append(tf.reshape(feature3, [200]))
                feature8ratio_label.append(int(id[2:]))




        database_feature, database_label = np.array(database_feature), np.array(database_label)
        feature2ratio, feauture4ratio, feature8ratio = np.array(feature2ratio), np.array(feature4ratio), np.array(feature8ratio)
        feature2ratio_label, feature4ratio_label, feature8_label = np.array(feature2ratio_label), np.array(feature4ratio_label), np.array(feature8ratio_label)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(database_feature, database_label)
        feature2ratio_pred = knn.predict(feature2ratio)
        feature4ratio_pred = knn.predict(feature4ratio)
        feature8ratio_pred = knn.predict(feature8ratio)

        feature2ratio_cm = confusion_matrix(feature2ratio_label, feature2ratio_pred)
        feature2ratio_acc = accuracy_score(feature2ratio_label, feature2ratio_pred)
        feature4ratio_cm = confusion_matrix(feature4ratio_label, feature4ratio_pred)
        feature4ratio_acc = accuracy_score(feature4ratio_label, feature4ratio_pred)
        feature8ratio_cm = confusion_matrix(feature8ratio_label, feature8ratio_pred)
        feature8ratio_acc = accuracy_score(feature8ratio_label, feature8ratio_pred)

        plt.subplots(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.title(f'2 Ratio Acc {str(feature2ratio_acc)[0:5]}')
        plt.imshow(feature2ratio_cm, cmap='jet')
        plt.subplot(2, 2, 2)
        plt.title(f'4 Ratio Acc {str(feature4ratio_acc)[0:5]}')
        plt.imshow(feature4ratio_cm, cmap='jet')
        plt.subplot(2, 2, 3)
        plt.title(f'8 Ratio Acc {str(feature8ratio_acc)[0:5]}')
        plt.imshow(feature8ratio_cm, cmap='jet')
        plt.show()

    def reg_repeat(self):
        def inversion(latent, low, database_z, database_zreg):
            latent_init = latent
            for step in range(1, 11):
                _, _, latent_reg = self.reg(latent)

                rec_loss, dis_loss = self.inversion(latent_init, latent_reg, low, database_z, database_zreg, ratio=1, grad=False)
                total_loss = rec_loss + dis_loss

                if step > 1 and (total_loss > inversion_total_loss):
                    latent_reg = latent
                    total_loss = inversion_total_loss

                gradient_latent, rec_loss, dis_loss = self.inversion(latent_init, latent_reg, low, database_z, database_zreg, ratio=1)

                for lr in [(0.3 * i) * (1 / step) for i in range(0, 11)]:
                    z_search = latent_reg - (lr * gradient_latent)
                    _, rec_loss, dis_loss = self.inversion(latent_init, z_search, low, database_z, database_zreg, ratio=1)
                    if (rec_loss + dis_loss) < total_loss:
                        inversion_total_loss = rec_loss + dis_loss
                        total_loss = inversion_total_loss
                        latent_opti = z_search

                latent = latent_opti
            return latent_opti

        path_train = '/disk2/bosen/Datasets/AR_train/'
        path_test = '/disk2/bosen/Datasets/AR_aligment_other/'
        # ID = ['ID19/m-023-3-1.bmp', 'ID27/m-040-1-2.bmp', 'ID29/m-044-3-0.bmp', 'ID32/m-047-3-0.bmp']
        ID = ['ID01/', 'ID02/', 'ID03/', 'ID04/']


        mPSNR = [[] for i in range(100)]
        mKNN = [0 for i in range(100)]
        mCLS = [0 for i in range(100)]

        # database_feature, database_label = [], []
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
        #             feature = self.encoder(image.reshape(1, 64, 64, 1))
        #             feature1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
        #             feature2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
        #             feature3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
        #
        #             _, _, feature = self.reg(feature)
        #             _, _, feature1 = self.reg(feature1)
        #             _, _, feature2 = self.reg(feature2)
        #             _, _, feature3 = self.reg(feature3)
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
        #             database_label.append(int(id[2:]) - 1)
        #             database_label.append(int(id[2:]) - 1)
        #             database_label.append(int(id[2:]) - 1)
        #             database_label.append(int(id[2:]) - 1)
        #
        # database_feature, database_label = np.array(database_feature), np.array(database_label)
        # knn = KNeighborsClassifier(n_neighbors=3)
        # knn.fit(database_feature, database_label)

        database_z, database_zreg = self.database()


        plt.subplots(figsize=(4, 12))
        plt.subplots_adjust(hspace=0, wspace=0)
        for id_num, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_test + id)):
                if num == 1:
                    break
                high_image = cv2.imread(path_test + id + filename, 0) / 255
                high_image = cv2.resize(high_image, (64, 64), cv2.INTER_CUBIC)
                test_image = high_image
                plt.subplot(12, 4, id_num+1)
                plt.axis('off')
                plt.imshow(tf.reshape(test_image, [64, 64]), cmap='gray')

                count = 0
                for step in range(1, 101):
                    print(step)
                    feature = self.encoder(tf.reshape(test_image, [1, 64, 64, 1]))
                    _, _, feature = self.reg(feature)
                    # feature = inversion(feature, high_image, database_z, database_zreg)
                    test_image = self.generator(feature)

                    mPSNR[step - 1].append(tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(test_image, dtype=tf.float32), max_val=1)[0])
                    # feature_knn = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
                    # pred_knn = knn.predict(feature_knn)
                    # _, pred_cls = self.cls(feature)
                    #
                    #
                    # if int(id[2:4]) - 1 == pred_knn:
                    #     mKNN[step-1] += 1
                    #     print(id)
                    # if int(id[2:4]) - 1 == np.argmax(pred_cls, axis=-1)[0]:
                    #     mCLS[step-1] += 1
                    #     print(id)

                    if (step == 1) or (step%10 == 0):
                        count += 1
                        plt.subplot(12, 4, (id_num + 1) + (count*4))
                        plt.axis('off')
                        plt.imshow(tf.reshape(test_image, [64, 64]), cmap='gray')

        plt.show()
        mPSNR = tf.reduce_mean(mPSNR, axis=-1)
        plt.plot(mPSNR, label='mPSNR')
        plt.title('mPSNR')
        plt.xlabel('Iterate times')
        plt.ylabel('mPSNR value')
        plt.legend()
        plt.show()


        plt.plot(mKNN, label='mKNN')
        plt.title('mKNN')
        plt.xlabel('Iterate times')
        plt.ylabel('number')
        plt.legend()
        plt.show()

        plt.plot(mCLS, label='mCLS')
        plt.title('mCLS')
        plt.xlabel('Iterate times')
        plt.ylabel('number')
        plt.legend()
        plt.show()

    def psnr_ssim(self, ratio, kernel, plot=False):
        z, zreg, zreg_opti, high_image, low_image, _ = self.reg_inversion(ratio, kernel)
        mPSNR, mSSIM = [[] for i in range(3)], [[] for i in range(3)]


        plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(wspace=0, hspace=0)
        count = 0
        for num, (latent, latent_reg, latent_reg_opti, high, low) in enumerate(zip(z, zreg, zreg_opti, high_image, low_image)):
            syn = self.generator(latent)
            syn_reg = self.generator(tf.reshape(latent_reg, [-1, 200]))
            syn_reg_opti = self.generator(tf.reshape(latent_reg_opti, [-1, 200]))
            if num > 7 and count <= 6:
                count += 1
                plt.subplot(7, 7, count)
                plt.axis('off')
                plt.imshow(tf.reshape(high, [64, 64]), cmap='gray')

                plt.subplot(7, 7, count+7)
                plt.axis('off')
                plt.imshow(tf.reshape(low, [64, 64]), cmap='gray')

                plt.subplot(7, 7, count+14)
                plt.axis('off')
                plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

                plt.subplot(7, 7, count+21)
                plt.axis('off')
                plt.imshow(tf.reshape(syn_reg, [64, 64]), cmap='gray')

                plt.subplot(7, 7, count+28)
                plt.axis('off')
                plt.imshow(tf.reshape(syn_reg_opti, [64, 64]), cmap='gray')

                plt.subplot(7, 7, count + 35)
                plt.axis('off')
                diff_syn_reg = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg, [64, 64]).numpy())
                diff_syn_reg_binary = (diff_syn_reg > np.mean(diff_syn_reg) + 0.5*np.std(diff_syn_reg)).astype(int)
                plt.imshow(tf.reshape(diff_syn_reg_binary, [64, 64]), cmap='gray')

                plt.subplot(7, 7, count + 42)
                plt.axis('off')
                diff = abs(tf.reshape(high, [64, 64]).numpy() - tf.reshape(syn_reg_opti, [64, 64]).numpy())
                diff_binary = (diff > np.mean(diff_syn_reg) + 0.5*np.std(diff_syn_reg)).astype(int)
                plt.imshow(tf.reshape(diff_binary, [64, 64]), cmap='gray')

            if plot:
                if count == 7:
                    count = 8
                    if self.testing == 'testing2':
                        plt.savefig(f'result/reg_test_gaussian_blur3/90id_var_{ratio}_ratio_result')
                        plt.close()
                    elif self.testing == 'testing3':
                        plt.savefig(f'result/reg_test_gaussian_blur3/21id_var_{ratio}_ratio_result')
                        plt.close()
            mPSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
            mPSNR[1].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
            mPSNR[2].append(tf.image.psnr(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])
            mSSIM[0].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
            mSSIM[1].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg, dtype=tf.float32), max_val=1)[0])
            mSSIM[2].append(tf.image.ssim(tf.cast(tf.reshape(high, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn_reg_opti, dtype=tf.float32), max_val=1)[0])

        print(mPSNR[0])
        print(mPSNR[1])
        print(mPSNR[2])
        mPSNR = tf.reduce_mean(mPSNR, axis=-1)
        mSSIM = tf.reduce_mean(mSSIM, axis=-1)
        print(mPSNR, mSSIM)



if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    reg_test = reg_test(testing='testing3')

    # reg_test.psnr_ssim(1, kernel=(1, 1))
    # reg_test.psnr_ssim(2, kernel=(3, 3))
    # reg_test.psnr_ssim(2, kernel=(7, 7))
    # reg_test.psnr_ssim(2, kernel=(11, 11))
    #
    # reg_test.psnr_ssim(3.2, kernel=(3, 3))
    # reg_test.psnr_ssim(3.2, kernel=(7, 7))
    # reg_test.psnr_ssim(3.2, kernel=(11, 11))

    reg_test.psnr_ssim(4, kernel=(3, 3))
    reg_test.psnr_ssim(4, kernel=(7, 7))
    reg_test.psnr_ssim(4, kernel=(11, 11))

    reg_test.psnr_ssim(6.4, kernel=(3, 3))
    reg_test.psnr_ssim(6.4, kernel=(7, 7))
    reg_test.psnr_ssim(6.4, kernel=(11, 11))

    reg_test.psnr_ssim(8, kernel=(3, 3))
    reg_test.psnr_ssim(8, kernel=(7, 7))
    reg_test.psnr_ssim(8, kernel=(11, 11))


    # reg_test.reg_repeat()
    # reg_test.cls_test(2)
    # reg_test.cls_test(4)
    # reg_test.cls_test(8)

    # reg_test.psnr_ssim(ratio=2)
    # reg_test.psnr_ssim(ratio=4)
    # reg_test.psnr_ssim(ratio=8)











