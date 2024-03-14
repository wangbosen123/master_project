import os
import cv2
import csv
from face_recognition import *
from build_model import *

def prepare_all_data():
    high_reso, low_reso = [], []
    zd_feature, zg_feature = [], []

    with open(f'result/inversion_result/zd_space_feature.csv', newline='') as zd_csvfile, open('result/inversion_result/zg_space_feature.csv', newline='') as zg_csvflie:
        zd_rows = csv.reader(zd_csvfile, quoting=csv.QUOTE_NONNUMERIC)
        zg_rows = csv.reader(zg_csvflie, quoting=csv.QUOTE_NONNUMERIC)
        rows = list(zip(zd_rows, zg_rows))
        for (zd, zg) in rows:
            zd_feature.append(zd)
            zg_feature.append(zg)
    zd_feature, zg_feature = np.array(zd_feature), np.array(zg_feature)

    path_AR = 'AR_aligment_test/'
    for id in os.listdir(path_AR):
        for filename in os.listdir(path_AR + id):
            if '-1-' in filename or '-14-' in filename:
                image = cv2.imread(path_AR + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                # low1_image = cv2.resize(image, (32, 32), cv2.INTER_CUBIC)
                # low2_image = cv2.resize(image, (16, 16), cv2.INTER_CUBIC)
                low3_image = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
                # low_reso.append(cv2.resize(low1_image, (64, 64), cv2.INTER_CUBIC))
                # low_reso.append(cv2.resize(low2_image, (64, 64), cv2.INTER_CUBIC))
                low_reso.append(cv2.resize(low3_image, (64, 64), cv2.INTER_CUBIC))
                high_reso.append(image)
    low_reso = np.array(low_reso).reshape(-1, 64, 64, 1)
    high_reso = np.array(high_reso).reshape(-1, 64, 64, 1)
    print(low_reso.shape, high_reso.shape)
    return high_reso, low_reso, zd_feature, zg_feature

# class GAN_Inversion_zg():
#     def __init__(self, epochs, learning_rate, type):
#         self.epochs = epochs
#         self.learning_rate = learning_rate
#         self.type = type
#         self.encoder = encoder()
#         self.ztozd = ZtoZd()
#         self.decoder = decoder()
#         self.ztozg = ZtoZg()
#         self.generator = generator()
#         self.discriminator = Patch_discriminator()
#         self.ID_cls = load_model('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ID_cls.h5')
#
#         self.checkpoint_encoder = tf.train.Checkpoint(self.encoder)
#         self.checkpoint_ztozd = tf.train.Checkpoint(self.ztozd)
#         self.checkpoint_decoder = tf.train.Checkpoint(self.decoder)
#         self.checkpoint_ztozg = tf.train.Checkpoint(self.ztozg)
#         self.checkpoint_generator = tf.train.Checkpoint(self.generator)
#         self.checkpoint_discriminator = tf.train.Checkpoint(self.discriminator)
#         self.checkpoint_encoder.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation")
#         self.checkpoint_ztozd.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
#         self.checkpoint_decoder.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/decoder_stage2_distillation')
#         self.checkpoint_ztozg.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
#         self.checkpoint_generator.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation")
#         self.checkpoint_discriminator.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/Patch_D")
#         self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#         self.test_high_reso, self.test_low_reso, self.zd_feature, self.zg_feature = prepare_all_data()
#         self.forward_zg_result = []
#         self.inversion_zd_result = []
#         for image in self.test_low_reso:
#             image = image.reshape(1, 64, 64, 1)
#             z, _ = self.encoder(image)
#             zg, _, _ = self.ztozg(z)
#             self.forward_zg_result.append(tf.reshape(zg, [200]))
#
#         with open(f'result/inversion_result/AR_test_low_inv/final_test_zd_L_inv.csv', newline='') as csvfile:
#             rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#             for num, feature in enumerate(rows):
#                 self.inversion_zd_result.append(feature)
#
#         inversion_zd_result = self.inversion_zd_result[0:123]
#         self.forward_zg_result = np.array(self.forward_zg_result)
#         self.inversion_zd_result = np.array(inversion_zd_result)
#         self.zd_neighbor, self.zg_neighbor = self.zd_inv_neighbor()
#         print(self.test_high_reso.shape, self.test_low_reso.shape, self.zd_feature.shape, self.zg_feature.shape)
#
#     def zd_inv_neighbor(self):
#         zd_neighbor = [[] for i in range(self.inversion_zd_result.shape[0])]
#         zg_neighbor = [[] for i in range(self.inversion_zd_result.shape[0])]
#         for num, zd in enumerate(self.inversion_zd_result):
#             zd = tf.reshape(zd, [1, 200])
#             zd = tf.tile(zd, [self.zd_feature.shape[0], 1])
#             distance = tf.Variable(tf.cast(tf.sqrt(tf.reduce_sum(tf.square(zd - self.zd_feature), axis=-1)), dtype=tf.float32), dtype=tf.float32)
#
#             #neighbor set the 15 images.
#             for i in range(3):
#                 min_index = tf.argmin(distance, 0)
#                 distance[min_index].assign(1000)
#                 zd_neighbor[num].append(self.zd_feature[min_index])
#                 zg_neighbor[num].append(self.zg_feature[min_index])
#         zd_neighbor = np.array(zd_neighbor)
#         zg_neighbor = np.array(zg_neighbor)
#         return zd_neighbor, zg_neighbor
#
#     def perceptual_loss(self, real_high, fake_high):
#         real_high, fake_high = tf.cast(real_high, dtype="float32"), tf.cast(fake_high, dtype="float32")
#         real_high = tf.image.grayscale_to_rgb(real_high)
#         fake_high = tf.image.grayscale_to_rgb(fake_high)
#         real_feature = self.feature_extraction(real_high)
#         fake_feature = self.feature_extraction(fake_high)
#         distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
#         return distance
#
#     def distillation_loss(self, zd_inv, zg_inv):
#         zd_inv, zg_inv = tf.reshape(zd_inv, [zd_inv.shape[0], 1, zd_inv.shape[1]]), tf.reshape(zg_inv, [zg_inv.shape[0], 1, zg_inv.shape[1]])
#         zd_inv = tf.tile(zd_inv, [1, self.zd_neighbor.shape[1], 1])
#         zg_inv = tf.tile(zg_inv, [1, self.zd_neighbor.shape[1], 1])
#         zd_distance = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(zd_inv - self.zd_neighbor), axis=-1)) / tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(zd_inv - self.zd_neighbor), axis=-1))), dtype=tf.float32)
#         zg_distance = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(zg_inv - self.zg_neighbor), axis=-1)) / tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(zg_inv - self.zg_neighbor), axis=-1))), dtype=tf.float32)
#         distillation_loss = tf.reduce_sum(tf.abs(zg_distance - zd_distance))
#         return distillation_loss
#
#     def zg_inversion_step(self, real_low, zd_inv, code, normalization=None):
#         real_low = real_low.reshape(-1, 64, 64, 1)
#         code = tf.cast(code, dtype=tf.float32)
#         with tf.GradientTape(persistent=True) as code_tape:
#             code_tape.watch(code)
#             fake_high = tf.reshape(self.generator(code), [-1, 64, 64, 1])
#
#             fake_pred = self.discriminator(fake_high)
#             GT_fake = tf.ones_like(fake_pred)
#             pred_L = self.ID_cls(zd_inv)
#             pred_H = self.ID_cls(code)
#
#             synthesis_image = []
#             for i in range(real_low.shape[0]):
#                 fake = tf.image.resize(fake_high[i], [8, 8], method='bicubic')
#                 # if i % 3 == 0:
#                 #     fake = tf.image.resize(fake_high[i-1], [8, 8], method='bicubic')
#                 # if i % 3 == 1:
#                 #     fake = tf.image.resize(fake_high[i-1], [32, 32], method='bicubic')
#                 # if i % 3 == 2:
#                 #     fake = tf.image.resize(fake_high[i-1], [16, 16], method='bicubic')
#                 synthesis_image.append(tf.image.resize(fake, [64, 64], method='bicubic'))
#             synthesis_image = tf.cast(synthesis_image, dtype=tf.float32)
#
#             rec_loss = tf.cast(tf.reduce_mean(tf.square(real_low - synthesis_image)), dtype=tf.float32)
#             style_loss = self.perceptual_loss(real_low, synthesis_image)
#             dis_loss = self.distillation_loss(zd_inv, code)
#             id_loss = tf.reduce_mean(tf.square(pred_L - pred_H))
#             adv_loss = tf.reduce_mean(tf.square(GT_fake - fake_pred))
#
#             if normalization is None:
#                 total_loss = rec_loss + style_loss + adv_loss
#
#             if normalization is not None:
#                 # normalized the rec_loss
#                 if rec_loss > normalization[0][0]:
#                     normalization[0][0] = rec_loss
#                 if rec_loss < normalization[0][1]:
#                     normalization[0][1] = rec_loss
#                 # normalized the style_loss
#                 if style_loss > normalization[1][0]:
#                     normalization[1][0] = style_loss
#                 if style_loss < normalization[1][1]:
#                     normalization[1][1] = style_loss
#                 # normalized the distillation_loss
#                 if dis_loss > normalization[2][0]:
#                     normalization[2][0] = dis_loss
#                 if dis_loss < normalization[2][1]:
#                     normalization[2][1] = dis_loss
#                 # normalized the ID loss
#                 if id_loss > normalization[3][0]:
#                     normalization[3][0] = id_loss
#                 if id_loss < normalization[3][1]:
#                     normalization[3][1] = id_loss
#                 # normalized the Adv loss
#                 if adv_loss > normalization[4][0]:
#                     normalization[4][0] = adv_loss
#                 if adv_loss < normalization[4][1]:
#                     normalization[4][1] = adv_loss
#
#                 rec_loss_normalization = 10 * ((rec_loss - normalization[0][1]) / (normalization[0][0] - normalization[0][1]))
#                 style_loss_normalization = ((style_loss - normalization[1][1]) / (normalization[1][0] - normalization[1][1]))
#                 adv_loss_normalization = ((adv_loss - normalization[4][1]) / (normalization[4][0] - normalization[4][1]))
#                 dis_loss_normalization = ((dis_loss - normalization[2][1]) / (normalization[2][0] - normalization[2][1]))
#                 id_loss_normalization = ((id_loss - normalization[3][1]) / (normalization[3][0] - normalization[3][1]))
#                 total_loss = rec_loss_normalization + style_loss_normalization + adv_loss_normalization
#
#
#                 # update the regression model
#             if normalization is None:
#                 gradient_code = code_tape.gradient(total_loss, code)
#                 code = code - self.learning_rate * gradient_code
#                 return rec_loss, style_loss, dis_loss, id_loss, adv_loss, code
#
#             if normalization is not None:
#                 gradient_code = code_tape.gradient(total_loss, code)
#                 code = code - self.learning_rate * gradient_code
#                 return rec_loss_normalization, style_loss_normalization, dis_loss_normalization, id_loss_normalization, adv_loss_normalization, code, normalization
#
#     def zg_inversion(self):
#         loss_normalization = [[] for i in range(5)]
#         rec_loss_epoch = []
#         style_loss_epoch = []
#         dis_loss_epoch = []
#         id_loss_epoch = []
#         adv_loss_epoch = []
#         rec_loss_epoch_nor = []
#         style_loss_epoch_nor = []
#         dis_loss_epoch_nor = []
#         id_loss_epoch_nor = []
#         adv_loss_epoch_nor = []
#         code = self.forward_zg_result
#         zd_inv = self.inversion_zd_result
#         res_rec_loss = 10
#         for epoch in range(1, self.epochs):
#             if epoch <= 100:
#                 start = time.time()
#                 rec_loss, style_loss, dis_loss, id_loss, adv_loss, code = self.zg_inversion_step(self.test_high_reso, zd_inv, code)
#                 rec_loss_epoch.append(rec_loss)
#                 style_loss_epoch.append(style_loss)
#                 dis_loss_epoch.append(dis_loss)
#                 id_loss_epoch.append(id_loss)
#                 adv_loss_epoch.append(adv_loss)
#                 print("______________________________________")
#                 print(f"the epoch is {epoch}")
#                 print(f"the mse_loss is {rec_loss}")
#                 print(f"the perceptual_loss is {style_loss}")
#                 print(f'the distilation loss is {dis_loss}')
#                 print(f'the id loss is {id_loss}')
#                 print(f'the adv loss is {adv_loss}')
#                 print(f"the new_code is {code[0][0:10]}")
#                 np.savetxt(f"result/inversion_result/AR_test_inv/final_test_zg_{self.type}_inv_result.csv", code, delimiter=",")
#                 print("the spend time is %s second" % (time.time() - start))
#
#             if epoch == 100:
#                 loss_normalization[0].append(max(rec_loss_epoch[0:100])), loss_normalization[0].append(min(rec_loss_epoch[0:100]))
#                 loss_normalization[1].append(max(style_loss_epoch[0:100])), loss_normalization[1].append(min(style_loss_epoch[0:100]))
#                 loss_normalization[2].append(max(dis_loss_epoch[0:100])), loss_normalization[2].append(min(dis_loss_epoch[0:100]))
#                 loss_normalization[3].append(max(id_loss_epoch[0:100])), loss_normalization[3].append(min(id_loss_epoch[0:100]))
#                 loss_normalization[4].append(max(adv_loss_epoch[0:100])), loss_normalization[4].append(min(adv_loss_epoch[0:100]))
#
#             # train the regression model using loss normalization
#             if epoch > 100:
#                 start = time.time()
#                 rec_loss, style_loss, dis_loss, id_loss, adv_loss, code = self.zg_inversion_step(self.test_low_reso, zd_inv, code, normalization=None)
#                 rec_loss_epoch_nor.append(rec_loss)
#                 style_loss_epoch_nor.append(style_loss)
#                 dis_loss_epoch_nor.append(dis_loss)
#                 id_loss_epoch_nor.append(id_loss)
#                 adv_loss_epoch_nor.append(adv_loss)
#                 print("______________________________________")
#                 print(f"the epoch is {epoch}")
#                 print(f"the mse_loss is {rec_loss}")
#                 print(f"the perceptual_loss is {style_loss}")
#                 print(f'the distilation loss is {dis_loss}')
#                 print(f'the id loss is {id_loss}')
#                 print(f'the adv loss is {adv_loss}')
#                 print(f"the new_code is {code[0][0:10]}")
#                 print("the spend time is %s second" % (time.time() - start))
#                 if rec_loss < res_rec_loss:
#                     res_rec_loss = rec_loss
#                     np.savetxt(f"result/inversion_result/AR_test_inv2/final_test_zg_{self.type}_inv_result.csv", code, delimiter=",")
#                 if (epoch + 1) % 100 == 0 or epoch - 1 == 0:
#                     num = 1
#                     res = 0
#                     plt.subplots(figsize=(8, 4))
#                     plt.subplots_adjust(wspace=0, hspace=0)
#                     for count, image in enumerate(self.test_high_reso):
#                         plt.subplot(4, 15, res + 1)
#                         plt.axis('off')
#                         plt.imshow(image, cmap='gray')
#
#                         plt.subplot(4, 15, res + 16)
#                         plt.axis('off')
#                         plt.imshow(tf.reshape(self.test_low_reso[count], [64, 64]), cmap='gray')
#
#                         plt.subplot(4, 15, res + 31)
#                         plt.axis('off')
#                         plt.imshow(tf.reshape(self.generator(tf.reshape(self.forward_zg_result[count], [1, 200])), [64, 64]), cmap='gray')
#
#                         plt.subplot(4, 15, res + 46)
#                         plt.axis('off')
#                         plt.imshow(tf.reshape(self.generator(tf.reshape(code[count], [1, 200])), [64, 64]), cmap='gray')
#                         res += 1
#                         if (res) % 15 == 0:
#                             plt.savefig(
#                                 f'result/inversion_result/AR_test_inv2/zg_{self.type}_inv_{num}_epoch_{epoch}.jpg')
#                             plt.close()
#                             plt.subplots(figsize=(8, 4))
#                             plt.subplots_adjust(wspace=0, hspace=0)
#                             num += 1
#                             res = 0
#                     plt.close()
#         # plt.plot(rec_loss_epoch_nor)
#         # plt.title("the rec_loss")
#         # plt.savefig(f"result/inversion_result/AR_test_inv2/test_{self.type}_rec_loss.jpg")
#         # plt.close()
#
#         # plt.plot(style_loss_epoch_nor)
#         # plt.title("the perceptual_loss")
#         # plt.savefig(f"result/inversion_result/AR_test_inv2/test_{self.type}_perceptual_loss.jpg")
#         # plt.close()
#         #
#         # plt.plot(id_loss_epoch_nor)
#         # plt.title("the id loss")
#         # plt.savefig(f"result/inversion_result/AR_test_inv2/test_{self.type}_id_loss_id.jpg")
#         # plt.close()
#         #
#         # plt.plot(dis_loss_epoch_nor)
#         # plt.title("the distilation loss")
#         # plt.savefig(f"result/inversion_result/AR_test_inv2/test_{self.type}_distillation_loss.jpg")
#         # plt.close()
#         #
#         # plt.plot(adv_loss_epoch_nor)
#         # plt.title("the adv loss")
#         # plt.savefig(f"result/inversion_result/AR_test_inv2/test_{self.type}_adv_loss.jpg")
#         # plt.close()
#         #
#         # plt.plot(rec_loss_epoch_nor)
#         # plt.plot(style_loss_epoch_nor)
#         # plt.plot(dis_loss_epoch_nor)
#         # plt.plot(id_loss_epoch_nor)
#         # plt.legend(['rec_loss', 'per_loss', 'dis_loss', 'id_loss'])
#         # plt.title("the total loss")
#         # plt.savefig(f"result/inversion_result/AR_test_inv2/test_{self.type}_total_loss.jpg")
#         # plt.close()

# class GAN_Inversion_zg():
#     def __init__(self, epochs, learning_rate, type):
#         self.epochs = epochs
#         self.learning_rate = learning_rate
#         self.type = type
#         self.encoder = encoder()
#         self.ztozd = ZtoZd()
#         self.decoder = decoder()
#         self.ztozg = ZtoZg()
#         self.generator = generator()
#         self.discriminator = Patch_discriminator()
#         self.ID_cls = load_model('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ID_cls.h5')
#
#         self.checkpoint_encoder = tf.train.Checkpoint(self.encoder)
#         self.checkpoint_ztozd = tf.train.Checkpoint(self.ztozd)
#         self.checkpoint_decoder = tf.train.Checkpoint(self.decoder)
#         self.checkpoint_ztozg = tf.train.Checkpoint(self.ztozg)
#         self.checkpoint_generator = tf.train.Checkpoint(self.generator)
#         self.checkpoint_discriminator = tf.train.Checkpoint(self.discriminator)
#         self.checkpoint_encoder.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation")
#         self.checkpoint_ztozd.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
#         self.checkpoint_decoder.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/decoder_stage2_distillation')
#         self.checkpoint_ztozg.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
#         self.checkpoint_generator.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation")
#         self.checkpoint_discriminator.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/Patch_D")
#         self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#         self.test_high_reso, self.test_low_reso, self.zd_feature, self.zg_feature = prepare_all_data()
#         self.forward_zg_result = []
#         self.inversion_zd_result = []
#         for image in self.test_low_reso:
#             image = image.reshape(1, 64, 64, 1)
#             z, _ = self.encoder(image)
#             zg, _, _ = self.ztozg(z)
#             self.forward_zg_result.append(tf.reshape(zg, [200]))
#
#         with open(f'result/inversion_result/AR_test_low_inv/final_test_zd_L_inv.csv', newline='') as csvfile:
#             rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#             for num, feature in enumerate(rows):
#                 self.inversion_zd_result.append(feature)
#
#         inversion_zd_result = self.inversion_zd_result[0:123]
#         self.forward_zg_result = np.array(self.forward_zg_result)
#         self.inversion_zd_result = np.array(inversion_zd_result)
#         self.zd_neighbor, self.zg_neighbor = self.zd_inv_neighbor()
#         print(self.test_high_reso.shape, self.test_low_reso.shape, self.zd_feature.shape, self.zg_feature.shape)
#
#     def zd_inv_neighbor(self):
#         zd_neighbor = [[] for i in range(self.inversion_zd_result.shape[0])]
#         zg_neighbor = [[] for i in range(self.inversion_zd_result.shape[0])]
#         for num, zd in enumerate(self.inversion_zd_result):
#             zd = tf.reshape(zd, [1, 200])
#             zd = tf.tile(zd, [self.zd_feature.shape[0], 1])
#             distance = tf.Variable(tf.cast(tf.sqrt(tf.reduce_sum(tf.square(zd - self.zd_feature), axis=-1)), dtype=tf.float32), dtype=tf.float32)
#
#             #neighbor set the 15 images.
#             for i in range(3):
#                 min_index = tf.argmin(distance, 0)
#                 distance[min_index].assign(1000)
#                 zd_neighbor[num].append(self.zd_feature[min_index])
#                 zg_neighbor[num].append(self.zg_feature[min_index])
#         zd_neighbor = np.array(zd_neighbor)
#         zg_neighbor = np.array(zg_neighbor)
#         return zd_neighbor, zg_neighbor
#
#     def perceptual_loss(self, real_high, fake_high):
#         real_high, fake_high = tf.cast(real_high, dtype="float32"), tf.cast(fake_high, dtype="float32")
#         real_high = tf.image.grayscale_to_rgb(real_high)
#         fake_high = tf.image.grayscale_to_rgb(fake_high)
#         real_feature = self.feature_extraction(real_high)
#         fake_feature = self.feature_extraction(fake_high)
#         distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
#         return distance
#
#     def distillation_loss(self, zd_inv, zg_inv):
#         zd_inv, zg_inv = tf.reshape(zd_inv, [zd_inv.shape[0], 1, zd_inv.shape[1]]), tf.reshape(zg_inv, [zg_inv.shape[0], 1, zg_inv.shape[1]])
#         zd_inv = tf.tile(zd_inv, [1, self.zd_neighbor.shape[1], 1])
#         zg_inv = tf.tile(zg_inv, [1, self.zd_neighbor.shape[1], 1])
#         zd_distance = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(zd_inv - self.zd_neighbor), axis=-1)) / tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(zd_inv - self.zd_neighbor), axis=-1))), dtype=tf.float32)
#         zg_distance = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(zg_inv - self.zg_neighbor), axis=-1)) / tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(zg_inv - self.zg_neighbor), axis=-1))), dtype=tf.float32)
#         distillation_loss = tf.reduce_sum(tf.abs(zg_distance - zd_distance))
#         return distillation_loss
#
#     def zg_inversion_step(self, real_low, zd_inv, code):
#         real_low = real_low.reshape(-1, 64, 64, 1)
#         code = tf.cast(code, dtype=tf.float32)
#         with tf.GradientTape(persistent=True) as code_tape:
#             code_tape.watch(code)
#             fake_high = tf.reshape(self.generator(code), [-1, 64, 64, 1])
#
#             fake_pred = self.discriminator(fake_high)
#             GT_fake = tf.ones_like(fake_pred)
#             pred_L = self.ID_cls(zd_inv)
#             pred_H = self.ID_cls(code)
#
#             synthesis_image = []
#             for i in range(real_low.shape[0]):
#                 fake = tf.image.resize(fake_high[i], [8, 8], method='bicubic')
#                 # if i % 3 == 0:
#                 #     fake = tf.image.resize(fake_high[i-1], [8, 8], method='bicubic')
#                 # if i % 3 == 1:
#                 #     fake = tf.image.resize(fake_high[i-1], [32, 32], method='bicubic')
#                 # if i % 3 == 2:
#                 #     fake = tf.image.resize(fake_high[i-1], [16, 16], method='bicubic')
#                 synthesis_image.append(tf.image.resize(fake, [64, 64], method='bicubic'))
#             synthesis_image = tf.cast(synthesis_image, dtype=tf.float32)
#
#             rec_loss = 10 * tf.cast(tf.reduce_mean(tf.square(real_low - synthesis_image)), dtype=tf.float32)
#             style_loss = self.perceptual_loss(real_low, synthesis_image)
#             dis_loss = self.distillation_loss(zd_inv, code)
#             id_loss = tf.reduce_mean(tf.square(pred_L - pred_H))
#             adv_loss = tf.reduce_mean(tf.square(GT_fake - fake_pred))
#
#             total_loss = rec_loss + style_loss + adv_loss
#             gradient_code = code_tape.gradient(total_loss, code)
#             code = code - self.learning_rate * gradient_code
#             return rec_loss, style_loss, dis_loss, id_loss, adv_loss, code
#
#     def zg_inversion(self):
#         rec_loss_epoch = []
#         style_loss_epoch = []
#         dis_loss_epoch = []
#         id_loss_epoch = []
#         adv_loss_epoch = []
#         code = self.forward_zg_result
#         zd_inv = self.inversion_zd_result
#         res_rec_loss = 10
#         for epoch in range(1, self.epochs):
#             start = time.time()
#             rec_loss, style_loss, dis_loss, id_loss, adv_loss, code = self.zg_inversion_step(self.test_low_reso, zd_inv, code)
#             rec_loss_epoch.append(rec_loss)
#             style_loss_epoch.append(style_loss)
#             dis_loss_epoch.append(dis_loss)
#             id_loss_epoch.append(id_loss)
#             adv_loss_epoch.append(adv_loss)
#             print("______________________________________")
#             print(f"the epoch is {epoch}")
#             print(f"the mse_loss is {rec_loss}")
#             print(f"the perceptual_loss is {style_loss}")
#             print(f'the distilation loss is {dis_loss}')
#             print(f'the id loss is {id_loss}')
#             print(f'the adv loss is {adv_loss}')
#             print(f"the new_code is {code[0][0:10]}")
#             print("the spend time is %s second" % (time.time() - start))
#             if rec_loss < res_rec_loss:
#                 res_rec_loss = rec_loss
#                 np.savetxt(f"result/inversion_result/AR_test_inv2/final_test_zg_{self.type}_inv_result.csv", code, delimiter=",")
#             if (epoch + 1) % 100 == 0 or epoch - 1 == 0:
#                 num = 1
#                 res = 0
#                 plt.subplots(figsize=(8, 4))
#                 plt.subplots_adjust(wspace=0, hspace=0)
#                 for count, image in enumerate(self.test_high_reso):
#                     plt.subplot(4, 15, res + 1)
#                     plt.axis('off')
#                     plt.imshow(image, cmap='gray')
#
#                     plt.subplot(4, 15, res + 16)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(self.test_low_reso[count], [64, 64]), cmap='gray')
#
#                     plt.subplot(4, 15, res + 31)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(self.generator(tf.reshape(self.forward_zg_result[count], [1, 200])), [64, 64]), cmap='gray')
#
#                     plt.subplot(4, 15, res + 46)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(self.generator(tf.reshape(code[count], [1, 200])), [64, 64]), cmap='gray')
#                     res += 1
#                     if (res) % 15 == 0:
#                         plt.savefig(
#                             f'result/inversion_result/AR_test_inv2/zg_{self.type}_inv_{num}_epoch_{epoch}.jpg')
#                         plt.close()
#                         plt.subplots(figsize=(8, 4))
#                         plt.subplots_adjust(wspace=0, hspace=0)
#                         num += 1
#                         res = 0
#                 plt.close()

# def store_inversion_image():
#     h_test, m_test, l_test = [], [], []
#     filename = ['final_test_zg_H_inv_result.csv', 'final_test_zg_M_inv_result.csv', 'final_test_zg_L_inv_result.csv']
#
#     for name in filename:
#         with open(f'result/inversion_result/AR_test_inv2/{name}', newline='') as csvfile:
#             rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#             for num, feature in enumerate(rows):
#                 if 'H' in name:
#                     h_test.append(feature)
#                 if 'M' in name:
#                     m_test.append(feature)
#                 if 'L' in name:
#                     l_test.append(feature)
#     h_test, m_test, l_test = np.array(h_test), np.array(m_test), np.array(l_test)
#     print(h_test.shape, m_test.shape, l_test.shape)
#
#     count = 0
#     path = 'AR_aligment_test/'
#     for id in os.listdir(path):
#         for filename in os.listdir(path + id):
#             if '-1-' in filename or '-14-' in filename:
#                 syn_Lh = generator(tf.reshape(h_test[count], [1, 200]))
#                 syn_Lh = syn_Lh.numpy().reshape(64, 64)
#                 syn_Lm = generator(tf.reshape(m_test[count], [1,200]))
#                 syn_Lm = syn_Lm.numpy().reshape(64, 64)
#                 syn_Ll = generator(tf.reshape(l_test[count], [1, 200]))
#                 syn_Ll = syn_Ll.numpy().reshape(64, 64)
#                 count += 1
#                 cv2.imwrite(f'AR_alignment_test_inv2/L_h/ID{int(id[2:])}/{filename}', syn_Lh*255)
#                 cv2.imwrite(f'AR_alignment_test_inv2/L_m/ID{int(id[2:])}/{filename}', syn_Lm*255)
#                 cv2.imwrite(f'AR_alignment_test_inv2/L_l/ID{int(id[2:])}/{filename}', syn_Ll*255)

class GAN_Inversion_zd():
    def __init__(self, epochs, learning_rate, type):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.type = type
        self.encoder = encoder()
        self.ztozd = ZtoZd()
        self.decoder = decoder()
        self.ztozg = ZtoZg()

        self.checkpoint_encoder = tf.train.Checkpoint(self.encoder)
        self.checkpoint_ztozd = tf.train.Checkpoint(self.ztozd)
        self.checkpoint_decoder = tf.train.Checkpoint(self.decoder)
        self.checkpoint_ztozg = tf.train.Checkpoint(self.ztozg)
        self.checkpoint_encoder.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation")
        self.checkpoint_ztozd.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZd_stage2_distillation')
        self.checkpoint_decoder.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/decoder_stage2_distillation')
        self.checkpoint_ztozg.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
        self.test_high_reso, self.test_low_reso, self.zd_feature, self.zg_feature = prepare_all_data()
        self.forward_zd_result = []
        for image in self.test_low_reso:
            image = image.reshape(1, 64, 64, 1)
            z, _ = self.encoder(image)
            zd, _ = self.ztozd(z)
            self.forward_zd_result.append(tf.reshape(zd, [200]))
        self.forward_zd_result = np.array(self.forward_zd_result)

    def perceptual_loss(self, real_high, fake_high):
        real_high, fake_high = tf.cast(real_high, dtype="float32"), tf.cast(fake_high, dtype="float32")
        real_high = tf.image.grayscale_to_rgb(real_high)
        fake_high = tf.image.grayscale_to_rgb(fake_high)
        real_feature = self.feature_extraction(real_high)
        fake_feature = self.feature_extraction(fake_high)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def zd_inversion_step(self, real_low, code, leraning_rate):
        real_low = real_low.reshape(-1, 64, 64, 1)
        code = tf.cast(code, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(code)
            fake_low = self.decoder(code)
            rec_loss = 1000 * tf.reduce_mean(tf.square(real_low - fake_low))
            perceptual_loss = 100 * self.perceptual_loss(real_low, fake_low)
            total_loss = rec_loss + perceptual_loss

        gradient_code = code_tape.gradient(total_loss, code)
        code = code - leraning_rate * gradient_code
        return rec_loss, perceptual_loss, code

    def zd_inversion(self):
        rec_loss_epoch, per_loss_epoch = [], []
        code = self.forward_zd_result
        learning_rate = self.learning_rate
        res_loss = 100
        for epoch in range(self.epochs):
            start = time.time()
            print(self.test_low_reso.shape, code.shape)
            rec_loss, perceptual_loss, code = self.zd_inversion_step(self.test_low_reso, code, learning_rate)
            rec_loss_epoch.append(rec_loss)
            per_loss_epoch.append(perceptual_loss)

            print("______________________________________")
            print(f"the epoch is {epoch+1}")
            print(f"the mse_loss is {rec_loss}")
            print(f"the perceptual_loss is {perceptual_loss}")
            print(f"the new_code is {code[0][0:10]}")
            print("the spend time is %s second" % (time.time() - start))
            print(f'the loss minminum is {res_loss}')
            print(f'the current learning rate is {learning_rate}')
            if rec_loss < res_loss:
                res_loss = rec_loss
                print('update the latent code')
                np.savetxt(f"result/inversion_result/AR_test_low_inv/final_test_zd_{self.type}_inv.csv", code, delimiter=",")

            if (epoch + 1) % 100 == 0 or epoch == 0:
                num = 1
                res = 0
                plt.subplots(figsize=(8, 4))
                plt.subplots_adjust(wspace=0, hspace=0)
                for count, image in enumerate(self.test_high_reso[0: 15]):
                    plt.subplot(4, 15, res + 1)
                    plt.axis('off')
                    plt.imshow(image, cmap='gray')

                    plt.subplot(4, 15, res + 16)
                    plt.axis('off')
                    plt.imshow(tf.reshape(self.test_low_reso[count], [64, 64]), cmap='gray')

                    plt.subplot(4, 15, res + 31)
                    plt.axis('off')
                    plt.imshow(tf.reshape(self.decoder(tf.reshape(self.forward_zd_result[count], [1, 200])), [64, 64]), cmap='gray')

                    plt.subplot(4, 15, res + 46)
                    plt.axis('off')
                    plt.imshow(tf.reshape(self.decoder(tf.reshape(code[count], [1, 200])), [64, 64]), cmap='gray')
                    res += 1
                    if (count + 1) % 15 == 0:
                        plt.savefig(f'result/inversion_result/AR_test_low_inv/zd__{self.type}_inv_{num}_epoch_{epoch}.jpg')
                        plt.close()
                        plt.subplots(figsize=(8, 4))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        num += 1
                        res = 0

        plt.plot(rec_loss_epoch)
        plt.title("the rec_loss")
        plt.savefig(f"result/inversion_result/AR_test_low_inv/test_rec_{self.type}_loss.jpg")
        plt.close()

        plt.plot(per_loss_epoch)
        plt.title("the perceptual_loss")
        plt.savefig(f"result/inversion_result/AR_test_low_inv/test_perceptual_{self.type}_loss.jpg")
        plt.close()


class GAN_Inversion_zg():
    def __init__(self, epochs, learning_rate, type):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.type = type
        self.encoder = encoder()
        self.ztozg = ZtoZg()
        self.generator = generator()
        self.discriminator = Patch_discriminator()

        self.checkpoint_encoder = tf.train.Checkpoint(self.encoder)
        self.checkpoint_ztozg = tf.train.Checkpoint(self.ztozg)
        self.checkpoint_generator = tf.train.Checkpoint(self.generator)
        self.checkpoint_discriminator = tf.train.Checkpoint(self.discriminator)
        self.checkpoint_encoder.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/encoder_stage2_distillation")
        self.checkpoint_ztozg.restore('/home/bosen/PycharmProjects/WGAN-GP/model_weight/ZtoZg_stage3_distillation')
        self.checkpoint_generator.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation")
        self.checkpoint_discriminator.restore("/home/bosen/PycharmProjects/WGAN-GP/model_weight/Patch_D")
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
        self.test_high_reso, self.test_low_reso = self.prepare_all_data()
        self.forward_zg_result = []
        for image in self.test_low_reso:
            image = image.reshape(1, 64, 64, 1)
            z, _ = self.encoder(image)
            zg, _, _ = self.ztozg(z)
            self.forward_zg_result.append(tf.reshape(zg, [200]))
        self.forward_zg_result = np.array(self.forward_zg_result)
        print(self.test_high_reso.shape, self.test_low_reso.shape)

    def prepare_all_data(self):
        high_reso, low_reso = [], []
        path_AR = 'cls_datasets/train_data_var_large/'
        for id in os.listdir(path_AR):
            for filename in os.listdir(path_AR + id):
                if 'syn' in filename:
                    continue
                image = cv2.imread(path_AR + id + '/' + filename, 0) / 255
                low_image = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
                low_reso.append(cv2.resize(low_image, (64, 64), cv2.INTER_CUBIC))
                high_reso.append(image)
        low_reso = np.array(low_reso).reshape(-1, 64, 64, 1)
        high_reso = np.array(high_reso).reshape(-1, 64, 64, 1)
        return high_reso, low_reso

    def perceptual_loss(self, real_high, fake_high):
        real_high, fake_high = tf.cast(real_high, dtype="float32"), tf.cast(fake_high, dtype="float32")
        real_high = tf.image.grayscale_to_rgb(real_high)
        fake_high = tf.image.grayscale_to_rgb(fake_high)
        real_feature = self.feature_extraction(real_high)
        fake_feature = self.feature_extraction(fake_high)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def zg_inversion_step(self, real_low, code):
        real_low = real_low.reshape(-1, 64, 64, 1)
        code = tf.cast(code, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as code_tape:
            code_tape.watch(code)
            fake_high = tf.reshape(self.generator(code), [-1, 64, 64, 1])
            fake_pred = self.discriminator(fake_high)
            GT_fake = tf.ones_like(fake_pred)

            synthesis_image = []
            for i in range(real_low.shape[0]):
                fake = tf.image.resize(fake_high[i], [8, 8], method='bicubic')
                synthesis_image.append(tf.image.resize(fake, [64, 64], method='bicubic'))
            synthesis_image = tf.cast(synthesis_image, dtype=tf.float32)

            rec_loss = 10 * tf.cast(tf.reduce_mean(tf.square(real_low - synthesis_image)), dtype=tf.float32)
            style_loss = self.perceptual_loss(real_low, synthesis_image)
            adv_loss = tf.reduce_mean(tf.square(GT_fake - fake_pred))

            total_loss = rec_loss + style_loss + adv_loss
            gradient_code = code_tape.gradient(total_loss, code)
            code = code - self.learning_rate * gradient_code
            return rec_loss, style_loss, adv_loss, code

    def zg_inversion(self):
        rec_loss_epoch = []
        style_loss_epoch = []
        adv_loss_epoch = []
        code = self.forward_zg_result
        res_rec_loss = 10
        for epoch in range(1, self.epochs):
            start = time.time()
            rec_loss, style_loss, adv_loss, code = self.zg_inversion_step(self.test_low_reso, code)
            rec_loss_epoch.append(rec_loss)
            style_loss_epoch.append(style_loss)
            adv_loss_epoch.append(adv_loss)
            print("______________________________________")
            print(f"the epoch is {epoch}")
            print(f"the mse_loss is {rec_loss}")
            print(f"the perceptual_loss is {style_loss}")
            print(f'the adv loss is {adv_loss}')
            print(f"the new_code is {code[0][0:10]}")
            print("the spend time is %s second" % (time.time() - start))
            if rec_loss < res_rec_loss:
                res_rec_loss = rec_loss
                np.savetxt(f"cls_datasets/inver_train_var_large_result/final_test_zg_{self.type}_inv_result.csv", code, delimiter=",")
            if (epoch + 1) % 100 == 0 or epoch - 1 == 0:
                num = 1
                res = 0
                plt.subplots(figsize=(8, 4))
                plt.subplots_adjust(wspace=0, hspace=0)
                for count, image in enumerate(self.test_high_reso):
                    plt.subplot(4, 15, res + 1)
                    plt.axis('off')
                    plt.imshow(image, cmap='gray')

                    plt.subplot(4, 15, res + 16)
                    plt.axis('off')
                    plt.imshow(tf.reshape(self.test_low_reso[count], [64, 64]), cmap='gray')

                    plt.subplot(4, 15, res + 31)
                    plt.axis('off')
                    plt.imshow(tf.reshape(self.generator(tf.reshape(self.forward_zg_result[count], [1, 200])), [64, 64]), cmap='gray')

                    plt.subplot(4, 15, res + 46)
                    plt.axis('off')
                    plt.imshow(tf.reshape(self.generator(tf.reshape(code[count], [1, 200])), [64, 64]), cmap='gray')
                    res += 1
                    if (res) % 15 == 0:
                        plt.savefig(f'cls_datasets/inver_train_var_large_result/zg_{self.type}_inv_{num}_epoch_{epoch}.jpg')
                        plt.close()
                        plt.subplots(figsize=(8, 4))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        num += 1
                        res = 0
                plt.close()

def store_inversion_image():
    path_AR = 'cls_datasets/train_data_var_large/'
    test32, test20, test16, test12, test8 = [], [], [], [], []
    filename = ['final_test_zg_32_inv_result.csv', "final_test_zg_20_inv_result.csv", 'final_test_zg_16_inv_result.csv', 'final_test_zg_12_inv_result.csv', 'final_test_zg_8_inv_result.csv']
    for name in filename:
        with open(f'cls_datasets/inver_train_var_large_result/{name}', newline='') as csvfile:
            rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for num, feature in enumerate(rows):
                if '32' in name:
                    test32.append(feature)
                if '20' in name:
                    test20.append(feature)
                if '16' in name:
                    test16.append(feature)
                if '12' in name:
                    test12.append(feature)
                if '8' in name:
                    test8.append(feature)
    test32, test20, test16, test12, test8 = np.array(test32), np.array(test20), np.array(test16), np.array(test12), np.array(test8)

    for num, id in enumerate(os.listdir(path_AR)):
        syn_32 = generator(tf.reshape(test32[num], [1, 200]))
        syn_32 = syn_32.numpy().reshape(64, 64)
        syn_20 = generator(tf.reshape(test20[num], [1, 200]))
        syn_20 = syn_20.numpy().reshape(64, 64)
        syn_16 = generator(tf.reshape(test16[num], [1, 200]))
        syn_16 = syn_16.numpy().reshape(64, 64)
        syn_12 = generator(tf.reshape(test12[num], [1, 200]))
        syn_12 = syn_12.numpy().reshape(64, 64)
        syn_8 = generator(tf.reshape(test8[num], [1, 200]))
        syn_8 = syn_8.numpy().reshape(64, 64)

        cv2.imwrite(f'cls_datasets/train_data_var_large/ID{int(id[2:])}/32_syn.jpg', syn_32 * 255)
        cv2.imwrite(f'cls_datasets/train_data_var_large/ID{int(id[2:])}/20_syn.jpg', syn_20 * 255)
        cv2.imwrite(f'cls_datasets/train_data_var_large/ID{int(id[2:])}/16_syn.jpg', syn_16 * 255)
        cv2.imwrite(f'cls_datasets/train_data_var_large/ID{int(id[2:])}/12_syn.jpg', syn_12 * 255)
        cv2.imwrite(f'cls_datasets/train_data_var_large/ID{int(id[2:])}/8_syn.jpg', syn_8 * 255)
    # for num, id in enumerate(os.listdir(path_AR)):
    #     syn_32_1 = generator(tf.reshape(test32[num*2], [1, 200]))
    #     syn_32_1 = syn_32_1.numpy().reshape(64, 64)
    #     syn_32_2 = generator(tf.reshape(test32[(num*2)+1], [1, 200]))
    #     syn_32_2 = syn_32_2.numpy().reshape(64, 64)
    #
    #     syn_20_1 = generator(tf.reshape(test20[num * 2], [1, 200]))
    #     syn_20_1 = syn_20_1.numpy().reshape(64, 64)
    #     syn_20_2 = generator(tf.reshape(test20[(num * 2) + 1], [1, 200]))
    #     syn_20_2 = syn_20_2.numpy().reshape(64, 64)
    #
    #     syn_16_1 = generator(tf.reshape(test16[(num*2)], [1, 200]))
    #     syn_16_1 = syn_16_1.numpy().reshape(64, 64)
    #     syn_16_2 = generator(tf.reshape(test16[(num*2)+1], [1, 200]))
    #     syn_16_2 = syn_16_2.numpy().reshape(64, 64)
    #
    #     syn_12_1 = generator(tf.reshape(test12[(num * 2)], [1, 200]))
    #     syn_12_1 = syn_12_1.numpy().reshape(64, 64)
    #     syn_12_2 = generator(tf.reshape(test12[(num * 2) + 1], [1, 200]))
    #     syn_12_2 = syn_12_2.numpy().reshape(64, 64)
    #
    #     syn_8_1 = generator(tf.reshape(test8[(num*2)], [1, 200]))
    #     syn_8_1 = syn_8_1.numpy().reshape(64, 64)
    #     syn_8_2 = generator(tf.reshape(test8[(num * 2)], [1, 200]))
    #     syn_8_2 = syn_8_2.numpy().reshape(64, 64)
    #     num += 1
    #
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/32_1syn.jpg', syn_32_1 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/32_2syn.jpg', syn_32_2 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/20_1syn.jpg', syn_20_1 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/20_2syn.jpg', syn_20_2 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/16_1syn.jpg', syn_16_1 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/16_2syn.jpg', syn_16_2 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/12_1syn.jpg', syn_12_1 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/12_2syn.jpg', syn_12_2 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/8_1syn.jpg', syn_8_1 * 255)
    #     cv2.imwrite(f'cls_datasets/cls_test_data/ID{int(id[2:])}/8_2syn.jpg', syn_8_2 * 255)


if __name__ == "__main__":
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.55
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    generator = generator()
    generator.load_weights("/home/bosen/PycharmProjects/WGAN-GP/model_weight/generator_stage3_distillation")
    store_inversion_image()

    # zg_inversion = GAN_Inversion_zg(1200, 2000, type='8')
    # zg_inversion.zg_inversion()



























