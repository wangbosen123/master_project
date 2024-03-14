import random

from overall_model import *
from sklearn.decomposition import PCA

# class Regression():
#     def __init__(self, epochs, batch_num, batch_size):
#         # set parameters
#         self.epochs = epochs
#         self.batch_num = batch_num
#         self.batch_size = batch_size
#
#         # set the model
#         self.encoder = normal_encoder()
#         self.ztozd = ZtoZd()
#         self.decoder = decoder()
#         self.ztozg = ZtoZg()
#         self.regression = regression_model_with_instance()
#         self.generator = generator()
#         self.discriminator = patch_discriminator()
#         self.encoder.load_weights('model_weight/AE_encoder')
#         self.ztozd.load_weights('model_weight/AE_ztozd')
#         self.decoder.load_weights('model_weight/AE_decoder')
#         self.ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
#         self.regression.load_weights('model_weight/regression_pre_train')
#         self.generator.load_weights('model_weight/zd_zg_distillation_generator')
#         self.discriminator.load_weights('model_weight/patch_d')
#         self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#
#         # set data.
#         self.zdHs, self.zdhs, self.zdms, self.zdls, self.zgHs, self.zghs, self.zgms, self.zgls, self.zghs_intepolation, self.zgms_intepolation, self.zgls_intepolation = self.AR_regression_training_data()
#         print(self.zdHs.shape, self.zdhs.shape, self.zdms.shape, self.zdls.shape, self.zgHs.shape, self.zghs.shape, self.zgms.shape, self.zgls.shape, self.zghs_intepolation.shape, self.zgms_intepolation.shape, self.zgls_intepolation.shape)
#
#     def AR_regression_training_data(self):
#         def get_all_feature():
#             path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
#
#             data_path, label = [], []
#             ID = [f'ID{i}' for i in range(1, 91)]
#
#             for num, id in enumerate(ID):
#                 for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
#                     if 20 <= count < 40:
#                         data_path.append(path_AR_syn_train + id + '/' + filename)
#                         label.append(int(id[2:]))
#
#             zgHs, zghs, zgms, zgls = [], [], [], []
#             zdHs, zdhs, zdms, zdls = [], [], [], []
#             for count, path in enumerate(data_path):
#                 image = cv2.imread(path, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#
#                 low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
#                 low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
#                 low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
#
#                 zH = self.encoder(image.reshape(1, 64, 64, 1))
#                 zh = self.encoder(low1_image)
#                 zm = self.encoder(low2_image)
#                 zl = self.encoder(low3_image)
#
#                 zdH, _ = self.ztozd(zH)
#                 zdh, _ = self.ztozd(zh)
#                 zdm, _ = self.ztozd(zm)
#                 zdl, _ = self.ztozd(zl)
#
#                 zgH, _, _ = self.ztozg(zH)
#                 zgh, _, _ = self.ztozg(zh)
#                 zgm, _, _ = self.ztozg(zm)
#                 zgl, _, _ = self.ztozg(zl)
#
#                 zdHs.append(tf.reshape(zdH, [200]))
#                 zdhs.append(tf.reshape(zdh, [200]))
#                 zdms.append(tf.reshape(zdm, [200]))
#                 zdls.append(tf.reshape(zdl, [200]))
#
#                 zgHs.append(tf.reshape(zgH, [200]))
#                 zghs.append(tf.reshape(zgh, [200]))
#                 zgms.append(tf.reshape(zgm, [200]))
#                 zgls.append(tf.reshape(zgl, [200]))
#
#             zdHs, zdhs, zdms, zdls = np.array(zdHs), np.array(zdhs), np.array(zdms), np.array(zdls)
#             zgHs, zghs, zgms, zgls = np.array(zgHs), np.array(zghs), np.array(zgms), np.array(zgls)
#             return zdHs, zdhs, zdms, zdls, zgHs, zghs, zgms, zgls
#
#         def find_negative_data(zghs, zgms, zgls):
#             def calculate_cosine_similarity(zgs):
#                 def set_diagonal_blocks_to_zero(matrix):
#                     n = matrix.shape[0]
#                     block_size = n // 90
#                     modified_matrix = np.copy(matrix)
#                     for i in range(90):
#                         start_row = i * block_size
#                         end_row = (i + 1) * block_size
#                         modified_matrix[start_row:end_row, start_row:end_row] = 0
#                     return modified_matrix
#
#                 similarity_matrix = tf.matmul(zgs, tf.transpose(zgs)) / tf.matmul(tf.norm(zgs, axis=1, keepdims=True), tf.norm(tf.transpose(zgs), axis=0, keepdims=True))
#                 similarity_matrix = set_diagonal_blocks_to_zero(similarity_matrix)
#                 return similarity_matrix
#
#             def find_max_cosine_similarity(similarity_matrix, num_neighbors=3):
#                 num_images = similarity_matrix.shape[0]
#                 max_similarity_indices = []
#                 for i in range(num_images):
#                     similarity_row = similarity_matrix[i]
#                     sorted_indices = np.argsort(similarity_row)[-num_neighbors:][::-1]
#                     max_similarity_indices.append(sorted_indices)
#                 return max_similarity_indices
#
#             zghs_negative_pair, zgms_negative_pair, zgls_negative_pair = [[] for i in range(zghs.shape[0])], [[] for i in range(zgms.shape[0])], [[] for i in range(zgls.shape[0])]
#
#             similarity_matrix_zghs = calculate_cosine_similarity(zghs)
#             similarity_matrix_zgms = calculate_cosine_similarity(zgms)
#             similarity_matrix_zgls = calculate_cosine_similarity(zgls)
#             max_cosine_similarity_zgh_index = find_max_cosine_similarity(similarity_matrix_zghs)
#             max_cosine_similarity_zgm_index = find_max_cosine_similarity(similarity_matrix_zgms)
#             max_cosine_similarity_zgl_index = find_max_cosine_similarity(similarity_matrix_zgls)
#             max_cosine_similarity_index = list(zip(max_cosine_similarity_zgh_index, max_cosine_similarity_zgm_index, max_cosine_similarity_zgl_index))
#
#             for num, (zghs_index, zgms_index, zgls_index) in enumerate(max_cosine_similarity_index):
#                 for i in range(zghs_index.shape[0]):
#                     zghs_negative_pair[num].append(zghs[zghs_index[i]])
#                 for i in range(zgms_index.shape[0]):
#                     zgms_negative_pair[num].append(zgms[zgms_index[i]])
#                 for i in range(zgls_index.shape[0]):
#                     zgls_negative_pair[num].append(zgls[zgls_index[i]])
#             zghs_negative_pair, zgms_negative_pair, zgls_negative_pair = np.array(zghs_negative_pair), np.array(zgms_negative_pair), np.array(zgls_negative_pair)
#             return zghs_negative_pair, zgms_negative_pair, zgls_negative_pair
#
#         def intepolation(zghs, zgms, zgls, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair):
#             zghs_intepolation, zgms_intepolation, zgls_intepolation = [[] for i in range(zghs.shape[0])], [[] for i in range(zgms.shape[0])], [[] for i in range(zgls.shape[0])]
#             zghs_expand, zgms_expand, zgls_expand = tf.tile(tf.reshape(zghs, [-1, 1, 200]), [1, 3, 1]), tf.tile(tf.reshape(zgms, [-1, 1, 200]), [1, 3, 1]), tf.tile(tf.reshape(zgls, [-1, 1, 200]), [1, 3, 1])
#
#             zg_data = list(zip(zghs_expand, zgms_expand, zgls_expand, zghs_negative_pair, zgms_negative_pair, zgls_negative_pair))
#             for num, (zgh, zgm, zgl, zgh_neg, zgm_neg, zgl_neg) in enumerate(zg_data):
#                 zghs_intepolation[num].append(zgh * 0.7 + zgh_neg * 0.3)
#                 zgms_intepolation[num].append(zgm * 0.7 + zgm_neg * 0.3)
#                 zgls_intepolation[num].append(zgl * 0.7 + zgl_neg * 0.3)
#             zghs_intepolation = np.array(zghs_intepolation).reshape(-1, 3, 200)
#             zgms_intepolation = np.array(zgms_intepolation).reshape(-1, 3, 200)
#             zgls_intepolation = np.array(zgls_intepolation).reshape(-1, 3, 200)
#             return zghs_intepolation, zgms_intepolation, zgls_intepolation
#
#         def validate_negative_data_distribution(zghs, zgms, zgls, zghs_negative_data, zgms_negative_data, zgls_negative_data):
#             for name, zg, zg_neg in zip(['Zgh', 'Zgm', 'Zgl'], [zghs, zgms, zgls], [zghs_negative_data, zgms_negative_data, zgls_negative_data]):
#                 pca = PCA(n_components=2)
#                 total_zg = pca.fit_transform(zg)
#                 zg = pca.transform(tf.reshape(zg[0], [1, 200]))
#                 zg_neg = pca.transform(tf.reshape(zg_neg[0], [-1, 200]))
#
#                 plt.figure(figsize=(7, 5))
#                 plt.style.use('ggplot')
#                 plt.xlabel('x-axis')
#                 plt.ylabel('y-axis')
#                 plt.title(f'{name}')
#                 plt.scatter(total_zg[:][:, 0], total_zg[:][:, 1], c='b', marker='o', s=25)
#                 plt.scatter(zg[:][:, 0], zg[:][:, 1], c='g', marker='o', s=25)
#                 plt.scatter(zg_neg[:][:, 0], zg_neg[:][:, 1], c='y', marker='o', s=25)
#                 plt.legend([f'total {name}', f'ID1 {name}', f'ID1 {name}-negative-data'], loc='upper left')
#                 plt.show()
#                 plt.close()
#
#         zdHs, zdhs, zdms, zdls, zgHs, zghs, zgms, zgls = get_all_feature()
#         zghs_negative_data, zgms_negative_data, zgls_negative_data = find_negative_data(zghs, zgms, zgls)
#         zghs_intepolation, zgms_intepolation, zgls_intepolation = intepolation(zghs, zgms, zgls, zghs_negative_data, zgms_negative_data, zgls_negative_data)
#         # validate_negative_data_distribution(zghs, zgms, zgls, zghs_negative_data, zgms_negative_data, zgls_negative_data)
#         return zdHs, zdhs, zdms, zdls, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation
#
#     def reg_loss(self, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation):
#         zregHs_error = tf.reduce_mean(tf.square(zgHs - zregHs))
#         zreghs_error = tf.reduce_mean(tf.square(zghs - zreghs))
#         zregms_error = tf.reduce_mean(tf.square(zgms - zregms))
#         zregls_error = tf.reduce_mean(tf.square(zgls - zregls))
#
#         zreghs_aug_error = tf.reduce_mean(tf.square(zghs_intepolation.reshape(-1, 200) - zreghs_intepolation))
#         zregms_aug_error = tf.reduce_mean(tf.square(zgms_intepolation.reshape(-1, 200) - zregms_intepolation))
#         zregls_aug_error = tf.reduce_mean(tf.square(zgls_intepolation.reshape(-1, 200) - zregls_intepolation))
#         reg_loss = (zregHs_error + zreghs_error + zregms_error + zregls_error + zreghs_aug_error + zregms_aug_error + zregls_aug_error) / 7
#         return reg_loss
#
#     def image_loss(self, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation):
#         zregHs_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zgHs, [-1, 200])) - self.generator(tf.reshape(zregHs, [-1, 200]))))
#         zreghs_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zghs, [-1, 200])) - self.generator(tf.reshape(zreghs, [-1, 200]))))
#         zregms_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zgms, [-1, 200])) - self.generator(tf.reshape(zregms, [-1, 200]))))
#         zregls_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zgls, [-1, 200])) - self.generator(tf.reshape(zregls, [-1, 200]))))
#
#         zreghs_aug_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zghs_intepolation, [-1, 200])) - self.generator(tf.reshape(zreghs_intepolation, [-1, 200]))))
#         zregms_aug_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zgms_intepolation, [-1, 200])) - self.generator(tf.reshape(zregms_intepolation, [-1, 200]))))
#         zregls_aug_error = tf.reduce_mean(tf.square(self.generator(tf.reshape(zgls_intepolation, [-1, 200])) - self.generator(tf.reshape(zregls_intepolation, [-1, 200]))))
#         image_loss = (zregHs_error + zreghs_error + zregms_error + zregls_error + zreghs_aug_error + zregms_aug_error + zregls_aug_error) / 7
#         return image_loss
#
#     def style_loss(self, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation):
#         def style_loss_subfunction(real, fake):
#             real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
#             real = tf.image.grayscale_to_rgb(real)
#             fake = tf.image.grayscale_to_rgb(fake)
#
#             real_feature = self.feature_extraction(real)
#             fake_feature = self.feature_extraction(fake)
#             distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
#             return distance
#
#         zregHs_error = style_loss_subfunction(self.generator(tf.reshape(zgHs, [-1, 200])) , self.generator(tf.reshape(zregHs, [-1, 200])))
#         zreghs_error = style_loss_subfunction(self.generator(tf.reshape(zghs, [-1, 200])) , self.generator(tf.reshape(zreghs, [-1, 200])))
#         zregms_error = style_loss_subfunction(self.generator(tf.reshape(zgms, [-1, 200])) , self.generator(tf.reshape(zregms, [-1, 200])))
#         zregls_error = style_loss_subfunction(self.generator(tf.reshape(zgls, [-1, 200])) , self.generator(tf.reshape(zregls, [-1, 200])))
#
#         zreghs_aug_error = style_loss_subfunction(self.generator(tf.reshape(zghs_intepolation, [-1, 200])), self.generator(tf.reshape(zreghs_intepolation, [-1, 200])))
#         zregms_aug_error = style_loss_subfunction(self.generator(tf.reshape(zgms_intepolation, [-1, 200])), self.generator(tf.reshape(zregms_intepolation, [-1, 200])))
#         zregls_aug_error = style_loss_subfunction(self.generator(tf.reshape(zgls_intepolation, [-1, 200])), self.generator(tf.reshape(zregls_intepolation, [-1, 200])))
#         image_loss = (zregHs_error + zreghs_error + zregms_error + zregls_error + zreghs_aug_error + zregms_aug_error + zregls_aug_error) / 7
#         return image_loss
#
#     def adv_loss(self, zregH, zregh, zregm, zregl, zregh_intepolation, zregm_intepolation, zregl_intepolation):
#         syn_H, syn_h, syn_m, syn_l = self.generator(zregH), self.generator(zregh), self.generator(zregm), self.generator(zregl)
#         syn_h_inte, syn_m_inte, syn_l_inte = self.generator(zregh_intepolation), self.generator(zregm_intepolation), self.generator(zregl_intepolation)
#         syn_H_score, syn_h_score, syn_m_score, syn_l_score = self.discriminator(syn_H), self.discriminator(syn_h), self.discriminator(syn_m), self.discriminator(syn_l)
#         syn_h_inte_score, syn_m_inte_score, syn_l_inte_score = self.discriminator(syn_h_inte), self.discriminator(syn_m_inte), self.discriminator(syn_l_inte)
#
#         adv_loss = tf.reduce_mean(tf.square(syn_H_score - 1)) + \
#                    tf.reduce_mean(tf.square(syn_h_score - 1)) + \
#                    tf.reduce_mean(tf.square(syn_m_score - 1)) + \
#                    tf.reduce_mean(tf.square(syn_l_score - 1)) + \
#                    tf.reduce_mean(tf.square(syn_h_inte_score - 1)) + \
#                    tf.reduce_mean(tf.square(syn_m_inte_score - 1)) + \
#                    tf.reduce_mean(tf.square(syn_l_inte_score - 1))
#         return adv_loss / 6
#
#
#     def get_batch_data(self, data, batch_idx, batch_size, image=False):
#         high_images = []
#         range_min = batch_idx * batch_size
#         range_max = (batch_idx + 1) * batch_size
#
#         if range_max > len(data):
#             range_max = len(data)
#         index = list(range(range_min, range_max))
#         train_data = [data[idx] for idx in index]
#
#         if image:
#             for path in train_data:
#                 image = cv2.imread(path, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 high_images.append(image)
#
#             high_images = np.array(high_images).reshape(-1, 64, 64, 1)
#             return high_images
#         else:
#             return np.array(train_data)
#
#     def train_step(self, zdHs, zdhs, zdms, zdls, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, opti):
#         with tf.GradientTape(persistent=True) as tape:
#
#             zregHs, zreghs, zregms, zregls = self.regression([zgHs, zdHs]), self.regression([zghs, zdhs]), self.regression([zgms, zdms]), self.regression([zgls, zdls])
#             zreghs_intepolation = self.regression([zghs_intepolation.reshape(-1, 200), tf.reshape([zdhs[i] for i in range(zdhs.shape[0]) for x in range(3)], [-1, 200])])
#             zregms_intepolation = self.regression([zgms_intepolation.reshape(-1, 200), tf.reshape([zdms[i] for i in range(zdms.shape[0]) for x in range(3)], [-1, 200])])
#             zregls_intepolation = self.regression([zgls_intepolation.reshape(-1, 200), tf.reshape([zdls[i] for i in range(zdls.shape[0]) for x in range(3)], [-1, 200])])
#
#             # zreghs_intepolation, zregms_intepolation, zregls_intepolation = self.regression(zghs_intepolation.reshape(-1, 200)), self.regression(zgms_intepolation.reshape(-1, 200)), self.regression(zgls_intepolation.reshape(-1, 200))
#             # zregh_neg, zregm_neg, zregl_neg = self.regression(zgh_neg.reshape(-1, 200)), self.regression(zgm_neg.reshape(-1, 200)), self.regression(zgl_neg.reshape(-1, 200))
#
#
#             reg_loss = self.reg_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation)
#             image_loss = 100 * self.image_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation)
#             style_loss = 100 * self.style_loss(zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation)
#             adv_loss = 10 * self.adv_loss(zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation)
#             contrast_loss = 0
#             total_loss = reg_loss + image_loss + style_loss + adv_loss
#
#         grads = tape.gradient(total_loss, self.regression.trainable_variables)
#         opti.apply_gradients(zip(grads, self.regression.trainable_variables))
#         return reg_loss, image_loss, style_loss, adv_loss, contrast_loss
#
#     def plot_image_AR(self, epoch, train=True):
#         if train:
#             path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_train/'
#         else:
#             path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
#         forward_psnr, forward_ssim = [[], [], []], [[], [], []]
#         reg_psnr, reg_ssim = [[], [], []], [[], [], []]
#
#         plt.subplots(figsize=(7, 10))
#         plt.subplots_adjust(hspace=0, wspace=0)
#         count = 0
#         for num_id, ID in enumerate(os.listdir(path_AR_syn_test)):
#             for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
#                 if train:
#                     name = '11_train'
#                 else:
#                     name = '11_test'
#
#                 if name in filename:
#                     image = cv2.imread(path_AR_syn_test + ID + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#
#                     low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#
#                     zh = self.encoder(low1_image.reshape(1, 64, 64, 1))
#                     zm = self.encoder(low2_image.reshape(1, 64, 64, 1))
#                     zl = self.encoder(low3_image.reshape(1, 64, 64, 1))
#
#                     zdh, _ = self.ztozd(zh)
#                     zdm, _ = self.ztozd(zm)
#                     zdl, _ = self.ztozd(zl)
#
#                     zgh, _, _ = self.ztozg(zh)
#                     zgm, _, _ = self.ztozg(zm)
#                     zgl, _, _ = self.ztozg(zl)
#
#                     zregh = self.regression([zgh, zdh])
#                     zregm = self.regression([zgm, zdm])
#                     zregl = self.regression([zgl, zdl])
#
#                     forward_zh_syn, forward_zm_syn, forward_zl_syn = self.generator(zgh), self.generator(zgm), self.generator(zgl)
#                     reg_zh_syn, reg_zm_syn, reg_zl_syn = self.generator(zregh), self.generator(zregm), self.generator(zregl)
#
#                     forward_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
#                     forward_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
#                     forward_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
#                     reg_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
#                     reg_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
#                     reg_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
#                     forward_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
#                     forward_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
#                     forward_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
#                     reg_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
#                     reg_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
#                     reg_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
#
#                     plt.subplot(10, 7, count + 1)
#                     plt.axis('off')
#                     plt.imshow(image, cmap='gray')
#                     plt.subplot(10, 7, count + 8)
#                     plt.axis('off')
#                     plt.imshow(low1_image, cmap='gray')
#                     plt.subplot(10, 7, count + 15)
#                     plt.axis('off')
#                     plt.imshow(low2_image, cmap='gray')
#                     plt.subplot(10, 7, count + 22)
#                     plt.axis('off')
#                     plt.imshow(low3_image, cmap='gray')
#                     plt.subplot(10, 7, count + 29)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(forward_zh_syn, [64, 64]), cmap='gray')
#                     plt.subplot(10, 7, count + 36)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(forward_zm_syn, [64, 64]), cmap='gray')
#                     plt.subplot(10, 7, count + 43)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(forward_zl_syn, [64, 64]), cmap='gray')
#                     plt.subplot(10, 7, count + 50)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(reg_zh_syn, [64, 64]), cmap='gray')
#                     plt.subplot(10, 7, count + 57)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(reg_zm_syn, [64, 64]), cmap='gray')
#                     plt.subplot(10, 7, count + 64)
#                     plt.axis('off')
#                     plt.imshow(tf.reshape(reg_zl_syn, [64, 64]), cmap='gray')
#                     count += 1
#                     if (count) % 7 == 0:
#                         if train:
#                             plt.savefig(f'result/regression/AR_fine_tune/train_{epoch}_{num_id}result')
#                             plt.close()
#                         else:
#                             plt.savefig(f'result/regression/AR_fine_tune/test_{epoch}_{num_id}result')
#                             plt.close()
#                         plt.subplots(figsize=(7, 10))
#                         plt.subplots_adjust(hspace=0, wspace=0)
#                         count = 0
#         plt.close()
#         forward_psnr, forward_ssim, reg_psnr, reg_ssim = np.array(forward_psnr), np.array(forward_ssim), np.array(reg_psnr), np.array(reg_ssim)
#         print(f'the mean psnr forward is {np.mean(forward_psnr)}')
#         print(f'the mean psnr reg is {np.mean(reg_psnr)}')
#         print(f'the mean ssim forward is {np.mean(forward_ssim)}')
#         print(f'the mean ssim reg is {np.mean(reg_ssim)}')
#         print('--------------------------------')
#         if np.mean(forward_psnr) < np.mean(reg_psnr):
#             self.regression.save_weights('model_weight/best_regression')
#
#         total_psnr = list(zip(forward_psnr, reg_psnr))
#         total_ssim = list(zip(forward_ssim, reg_ssim))
#         for num, (forward, reg) in enumerate(total_psnr):
#             plt.plot(forward, marker='.')
#             plt.plot(reg, marker='.')
#
#             plt.legend([f'mPSNR={str(np.mean(forward))[0:5]}', f'mPSNR={str(np.mean(reg))[0:5]}'], loc='upper right')
#             plt.savefig(f'result/regression/AR_fine_tune/{epoch}-{2**(num + 1)}-ratio-PSNR')
#             plt.close()
#
#         for num, (forward, reg) in enumerate(total_ssim):
#             plt.plot(forward, marker='.')
#             plt.plot(reg, marker='.')
#             plt.legend([f'mSSIM={str(np.mean(forward))[0:5]}', f'mSSIM={str(np.mean(reg))[0:5]}'], loc='upper right')
#             plt.savefig(f'result/regression/AR_fine_tune/{epoch}-{2**(num+1)}-ratio-SSIM')
#             plt.close()
#
#     def main(self):
#         image_loss_epoch = []
#         style_loss_epoch = []
#         reg_loss_epoch = []
#         adv_loss_epoch = []
#         constrast_loss_epoch = []
#
#         data = list(zip(self.zdHs, self.zdhs, self.zdms, self.zdls, self.zgHs, self.zghs, self.zgms, self.zgls, self.zghs_intepolation, self.zgms_intepolation, self.zgls_intepolation))
#         np.random.shuffle(data)
#         data = list(zip(*data))
#
#         zdHs, zdhs, zdms, zdls, zgHs, zghs, zgms, zgls, zghs_intepolation, zgms_intepolation, zgls_intepolation = \
#         np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), np.array(data[4]), np.array(data[5]), np.array(data[6]), np.array(data[7]), np.array(data[8]), np.array(data[9]), np.array(data[10])
#
#
#         opti = tf.keras.optimizers.Adam(1e-3)
#         for epoch in range(11, self.epochs + 1):
#             start = time.time()
#             image_loss_batch = []
#             style_loss_batch = []
#             reg_loss_batch = []
#             adv_loss_batch = []
#             constrast_loss_batch = []
#
#             for batch in range(self.batch_num):
#                 batch_zdH = self.get_batch_data(zdHs, batch, self.batch_size)
#                 batch_zdh = self.get_batch_data(zdhs, batch, self.batch_size)
#                 batch_zdm = self.get_batch_data(zdms, batch, self.batch_size)
#                 batch_zdl = self.get_batch_data(zdls, batch, self.batch_size)
#                 batch_zgH = self.get_batch_data(zgHs, batch, self.batch_size)
#                 batch_zgh = self.get_batch_data(zghs, batch, self.batch_size)
#                 batch_zgm = self.get_batch_data(zgms, batch, self.batch_size)
#                 batch_zgl = self.get_batch_data(zgls, batch, self.batch_size)
#
#                 batch_zgh_intepolation = self.get_batch_data(zghs_intepolation, batch, self.batch_size)
#                 batch_zgm_intepolation = self.get_batch_data(zgms_intepolation, batch, self.batch_size)
#                 batch_zgl_intepolation = self.get_batch_data(zgls_intepolation, batch, self.batch_size)
#
#
#                 reg_loss, image_loss, style_loss, adv_loss, constrast_loss = \
#                 self.train_step(batch_zdH, batch_zdh, batch_zdm, batch_zdl, batch_zgH, batch_zgh, batch_zgm, batch_zgl,
#                                 batch_zgh_intepolation, batch_zgm_intepolation, batch_zgl_intepolation, opti)
#                 image_loss_batch.append(image_loss)
#                 style_loss_batch.append(style_loss)
#                 reg_loss_batch.append(reg_loss)
#                 adv_loss_batch.append(adv_loss)
#                 constrast_loss_batch.append(constrast_loss)
#
#             image_loss_epoch.append(np.mean(image_loss_batch))
#             style_loss_epoch.append(np.mean(style_loss_batch))
#             reg_loss_epoch.append(np.mean(reg_loss_batch))
#             adv_loss_epoch.append(np.mean(adv_loss_batch))
#             constrast_loss_epoch.append(np.mean(constrast_loss_batch))
#             print('------------')
#             print(f'the epoch is {epoch}')
#             print(f'the image_loss is {image_loss_epoch[-1]}')
#             print(f'the style_loss is {style_loss_epoch[-1]}')
#             print(f'the reg_loss is {reg_loss_epoch[-1]}')
#             print(f'the adv_loss is {adv_loss_epoch[-1]}')
#             print(f'the constrast_loss is {constrast_loss_epoch[-1]}')
#             print(f'the spend time is {time.time() - start} second')
#
#             print('------------------------------------------------')
#             # self.regression.save_weights('model_weight/regression_test')
#             self.plot_image_AR(epoch, train=False)
#             filename = 'AR_fine_tune'
#
#             # plt.plot(image_loss_epoch)
#             # plt.savefig(f'result/regression/{filename}/image_loss')
#             # plt.close()
#             #
#             # plt.plot(style_loss_epoch)
#             # plt.savefig(f'result/regression/{filename}/style_loss')
#             # plt.close()
#             #
#             # plt.plot(reg_loss_epoch)
#             # plt.savefig(f'result/regression/{filename}/reg_loss')
#             # plt.close()
#             #
#             # plt.plot(adv_loss_epoch)
#             # plt.savefig(f'result/regression/{filename}/adv_loss')
#             # plt.close()
#             #
#             # plt.plot(constrast_loss_epoch)
#             # plt.savefig(f'result/regression/{filename}/constrast_loss')
#             # plt.close()


class Regression():
    def __init__(self, epochs, batch_num, batch_size):
        # set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        # set the model
        self.encoder = normal_encoder()
        self.ztozd = ZtoZd()
        self.decoder = decoder()
        self.ztozg = ZtoZg()
        self.regression = regression_model_with_instance()
        self.generator = generator()
        self.discriminator = patch_discriminator()
        self.encoder.load_weights('model_weight/AE_encoder')
        self.ztozd.load_weights('model_weight/AE_ztozd')
        self.decoder.load_weights('model_weight/AE_decoder')
        self.ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
        self.regression.load_weights('model_weight/regression_pre_train')
        self.generator.load_weights('model_weight/zd_zg_distillation_generator')
        self.discriminator.load_weights('model_weight/patch_d')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        # set data.
        self.zdH, self.zdh, self.zdm, self.zdl, self.zgH, self.zgh, self.zgm, self.zgl, self.train_path = self.prepare_data()
        print(self.zdH.shape, self.zdh.shape, self.zdm.shape, self.zgH.shape, self.zgh.shape, self.zgm.shape, self.zgl.shape, self.train_path.shape)

    def prepare_data(self):
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        train_path = []
        ID = [f'ID{i}' for i in range(1, 91)]
        zd0s, zd1s, zd2s, zd3s, zg0s, zg1s, zg2s, zg3s = [], [], [], [], [], [], [], []
        for num, id in enumerate(ID):
            for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if 20 < count <= 40:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                    image = cv2.imread(path_AR_syn_train + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    z0, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
                    zd0, _ = self.ztozd(z0)
                    zd1, _ = self.ztozd(z1)
                    zd2, _ = self.ztozd(z2)
                    zd3, _ = self.ztozd(z3)

                    zg0, _, _ = self.ztozg(z0)
                    zg1, _, _ = self.ztozg(z1)
                    zg2, _, _ = self.ztozg(z2)
                    zg3, _, _ = self.ztozg(z3)

                    zd0s.append(tf.reshape(zd0, [200])), zd1s.append(tf.reshape(zd1, [200])), zd2s.append(tf.reshape(zd2, [200])), zd3s.append(tf.reshape(zd3, [200]))
                    zg0s.append(tf.reshape(zg0, [200])), zg1s.append(tf.reshape(zg1, [200])), zg2s.append(tf.reshape(zg2, [200])), zg3s.append(tf.reshape(zg3, [200]))
        zd0s, zd1s, zd2s, zd3s, zg0s, zg1s, zg2s, zg3s = np.array(zd0s), np.array(zd1s), np.array(zd2s), np.array(zd3s), np.array(zg0s), np.array(zg1s), np.array(zg2s), np.array(zg3s)
        train_path = np.array(train_path)
        return zd0s, zd1s, zd2s, zd3s, zg0s, zg1s, zg2s, zg3s, train_path

    def get_batch_data(self, data, batch_idx, batch_size, image=False):
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
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                images.append(image)
            return np.array(images).reshape(-1, 64, 64, 1)

        return np.array(train_data)

    def reg_loss(self, zgH, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation):
        zregHs_error = tf.reduce_mean(tf.square(zgH - zregHs))
        zreghs_error = tf.reduce_mean(tf.square(zgH - zreghs))
        zregms_error = tf.reduce_mean(tf.square(zgH - zregms))
        zregls_error = tf.reduce_mean(tf.square(zgH - zregls))

        zreghs_aug_error = tf.reduce_mean(tf.square(tf.reshape([zgH[i] for i in range(zgH.shape[0]) for x in range(3)], [-1, 200]) - zreghs_intepolation))
        zregms_aug_error = tf.reduce_mean(tf.square(tf.reshape([zgH[i] for i in range(zgH.shape[0]) for x in range(3)], [-1, 200]) - zregms_intepolation))
        zregls_aug_error = tf.reduce_mean(tf.square(tf.reshape([zgH[i] for i in range(zgH.shape[0]) for x in range(3)], [-1, 200]) - zregls_intepolation))
        reg_loss = (zregHs_error + zreghs_error + zregms_error + zregls_error + zreghs_aug_error + zregms_aug_error + zregls_aug_error) / 7
        return reg_loss

    def image_loss(self, high_images, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation):
        zregHs_error = tf.reduce_mean(tf.square(high_images - self.generator(tf.reshape(zregHs, [-1, 200]))))
        zreghs_error = tf.reduce_mean(tf.square(high_images - self.generator(tf.reshape(zreghs, [-1, 200]))))
        zregms_error = tf.reduce_mean(tf.square(high_images - self.generator(tf.reshape(zregms, [-1, 200]))))
        zregls_error = tf.reduce_mean(tf.square(high_images - self.generator(tf.reshape(zregls, [-1, 200]))))

        zreghs_aug_error = tf.reduce_mean(tf.square(tf.cast(tf.reshape([high_images[i] for i in range(high_images.shape[0]) for x in range(3)], [-1, 64, 64, 1]), dtype=tf.float32) - self.generator(tf.reshape(zreghs_intepolation,[-1, 200]))))
        zregms_aug_error = tf.reduce_mean(tf.square(tf.cast(tf.reshape([high_images[i] for i in range(high_images.shape[0]) for x in range(3)], [-1, 64, 64, 1]), dtype=tf.float32) - self.generator(tf.reshape(zregms_intepolation,[-1, 200]))))
        zregls_aug_error = tf.reduce_mean(tf.square(tf.cast(tf.reshape([high_images[i] for i in range(high_images.shape[0]) for x in range(3)], [-1, 64, 64, 1]), dtype=tf.float32) - self.generator(tf.reshape(zregls_intepolation,[-1, 200]))))
        image_loss = (zregHs_error + zreghs_error + zregms_error + zregls_error + zreghs_aug_error + zregms_aug_error + zregls_aug_error) / 7
        return image_loss

    def style_loss(self, high_images, zregHs, zreghs, zregms, zregls, zreghs_intepolation, zregms_intepolation, zregls_intepolation):
        def style_loss_subfunction(real, fake):
            real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
            real = tf.image.grayscale_to_rgb(real)
            fake = tf.image.grayscale_to_rgb(fake)

            real_feature = self.feature_extraction(real)
            fake_feature = self.feature_extraction(fake)
            distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
            return distance

        zregHs_error = style_loss_subfunction(high_images, self.generator(tf.reshape(zregHs, [-1, 200])))
        zreghs_error = style_loss_subfunction(high_images, self.generator(tf.reshape(zreghs, [-1, 200])))
        zregms_error = style_loss_subfunction(high_images, self.generator(tf.reshape(zregms, [-1, 200])))
        zregls_error = style_loss_subfunction(high_images, self.generator(tf.reshape(zregls, [-1, 200])))

        zreghs_aug_error = style_loss_subfunction(tf.cast(tf.reshape([high_images[i] for i in range(high_images.shape[0]) for x in range(3)], [-1, 64, 64, 1]), dtype=tf.float32), self.generator(tf.reshape(zreghs_intepolation, [-1, 200])))
        zregms_aug_error = style_loss_subfunction(tf.cast(tf.reshape([high_images[i] for i in range(high_images.shape[0]) for x in range(3)], [-1, 64, 64, 1]), dtype=tf.float32), self.generator(tf.reshape(zregms_intepolation, [-1, 200])))
        zregls_aug_error = style_loss_subfunction(tf.cast(tf.reshape([high_images[i] for i in range(high_images.shape[0]) for x in range(3)], [-1, 64, 64, 1]), dtype=tf.float32), self.generator(tf.reshape(zregls_intepolation, [-1, 200])))

        style_loss = (zregHs_error + zreghs_error + zregms_error + zregls_error + zreghs_aug_error + zregms_aug_error + zregls_aug_error) / 7
        return style_loss

    def adv_loss(self, zregH, zregh, zregm, zregl, zregh_intepolation, zregm_intepolation, zregl_intepolation):
        syn_H, syn_h, syn_m, syn_l = self.generator(zregH), self.generator(zregh), self.generator(zregm), self.generator(zregl)
        syn_h_inte, syn_m_inte, syn_l_inte = self.generator(zregh_intepolation), self.generator(zregm_intepolation), self.generator(zregl_intepolation)
        syn_H_score, syn_h_score, syn_m_score, syn_l_score = self.discriminator(syn_H), self.discriminator(syn_h), self.discriminator(syn_m), self.discriminator(syn_l)
        syn_h_inte_score, syn_m_inte_score, syn_l_inte_score = self.discriminator(syn_h_inte), self.discriminator(syn_m_inte), self.discriminator(syn_l_inte)

        adv_loss = tf.reduce_mean(tf.square(syn_H_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_h_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_m_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_l_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_h_inte_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_m_inte_score - 1)) + \
                   tf.reduce_mean(tf.square(syn_l_inte_score - 1))
        return adv_loss / 6

    def train_step(self, high_images, zdH, zdh, zdm, zdl, zgH, zgh, zgm, zgl, opti):
        with tf.GradientTape(persistent=True) as tape:

            def intepolation(zgH, zgh, zgm, zgl):
                zgh_intepolation, zgm_intepolation, zgl_intepolation = [[] for i in range(zgh.shape[0])], [[] for i in range(zgm.shape[0])], [[] for i in range(zgl.shape[0])]

                for num in range(zgh.shape[0]):
                    for i in [0.2, 0.5, 0.8]:
                        zgh_intepolation[num].append(zgH[num] * i + zgh[num] * (1-i))
                        zgm_intepolation[num].append(zgH[num] * i + zgm[num] * (1-i))
                        zgl_intepolation[num].append(zgH[num] * i + zgl[num] * (1-i))

                return np.array(zgh_intepolation), np.array(zgm_intepolation), np.array(zgl_intepolation)


            zgh_intepolation, zgm_intepolation, zgl_intepolation = intepolation(zgH, zgh, zgm, zgl)


            zregH, zregh, zregm, zregl = self.regression([zgH, zdH]), self.regression([zgh, zdh]), self.regression([zgm, zdm]), self.regression([zgl, zdl])
            zregh_intepolation = self.regression([tf.reshape(zgh_intepolation, [-1, 200]),  tf.reshape([zdh[i] for i in range(zdh.shape[0]) for x in range(3)], [-1, 200])])
            zregm_intepolation = self.regression([tf.reshape(zgm_intepolation, [-1, 200]),  tf.reshape([zdm[i] for i in range(zdm.shape[0]) for x in range(3)], [-1, 200])])
            zregl_intepolation = self.regression([tf.reshape(zgl_intepolation, [-1, 200]),  tf.reshape([zdl[i] for i in range(zdl.shape[0]) for x in range(3)], [-1, 200])])
            reg_loss = self.reg_loss(zgH, zregH, zregh, zregm, zregl, zregh_intepolation, zregm_intepolation, zregl_intepolation)
            image_loss = 100 * self.image_loss(high_images, zregH, zregh, zregm, zregl, zregh_intepolation, zregm_intepolation, zregl_intepolation)
            style_loss = 50 * self.style_loss(high_images, zregH, zregh, zregm, zregl, zregh_intepolation, zregm_intepolation, zregl_intepolation)
            adv_loss = 20 * self.adv_loss(zregH, zregh, zregm, zregl, zregh_intepolation, zregm_intepolation, zregl_intepolation)
            contrast_loss = 0
            total_loss = reg_loss + image_loss + style_loss + adv_loss

        grads = tape.gradient(total_loss, self.regression.trainable_variables)
        opti.apply_gradients(zip(grads, self.regression.trainable_variables))
        return reg_loss, image_loss, style_loss, adv_loss, contrast_loss

    def plot_image_AR(self, epoch, train=True):
        if train:
            path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        else:
            path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'
        forward_psnr, forward_ssim = [[], [], []], [[], [], []]
        reg_psnr, reg_ssim = [[], [], []], [[], [], []]

        plt.subplots(figsize=(7, 10))
        plt.subplots_adjust(hspace=0, wspace=0)
        count = 0
        for num_id, ID in enumerate(os.listdir(path_AR_syn_test)):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if train:
                    name = '11_train'
                else:
                    name = '11_test'

                if name in filename:
                    image = cv2.imread(path_AR_syn_test + ID + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    zh = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    zm = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    zl = self.encoder(low3_image.reshape(1, 64, 64, 1))

                    zdh, _ = self.ztozd(zh)
                    zdm, _ = self.ztozd(zm)
                    zdl, _ = self.ztozd(zl)

                    zgh, _, _ = self.ztozg(zh)
                    zgm, _, _ = self.ztozg(zm)
                    zgl, _, _ = self.ztozg(zl)

                    zregh = self.regression([zgh, zdh])
                    zregm = self.regression([zgm, zdm])
                    zregl = self.regression([zgl, zdl])

                    forward_zh_syn, forward_zm_syn, forward_zl_syn = self.generator(zgh), self.generator(zgm), self.generator(zgl)
                    reg_zh_syn, reg_zm_syn, reg_zl_syn = self.generator(zregh), self.generator(zregm), self.generator(zregl)

                    forward_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                    forward_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                    forward_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                    reg_psnr[0].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                    reg_psnr[1].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                    reg_psnr[2].append(tf.image.psnr(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])
                    forward_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zh_syn, max_val=1)[0])
                    forward_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zm_syn, max_val=1)[0])
                    forward_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), forward_zl_syn, max_val=1)[0])
                    reg_ssim[0].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zh_syn, max_val=1)[0])
                    reg_ssim[1].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zm_syn, max_val=1)[0])
                    reg_ssim[2].append(tf.image.ssim(tf.reshape(tf.cast(image, dtype=tf.float32), [1, 64, 64, 1]), reg_zl_syn, max_val=1)[0])

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
                    plt.imshow(tf.reshape(forward_zh_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 36)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zm_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 43)
                    plt.axis('off')
                    plt.imshow(tf.reshape(forward_zl_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 50)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zh_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 57)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zm_syn, [64, 64]), cmap='gray')
                    plt.subplot(10, 7, count + 64)
                    plt.axis('off')
                    plt.imshow(tf.reshape(reg_zl_syn, [64, 64]), cmap='gray')
                    count += 1
                    if (count) % 7 == 0:
                        if train:
                            plt.savefig(f'result/regression/AR_fine_tune/train_{epoch}_{num_id}result')
                            plt.close()
                        else:
                            plt.savefig(f'result/regression/AR_fine_tune/test_{epoch}_{num_id}result')
                            plt.close()
                        plt.subplots(figsize=(7, 10))
                        plt.subplots_adjust(hspace=0, wspace=0)
                        count = 0
        plt.close()
        forward_psnr, forward_ssim, reg_psnr, reg_ssim = np.array(forward_psnr), np.array(forward_ssim), np.array(reg_psnr), np.array(reg_ssim)
        print(f'the mean psnr forward is {np.mean(forward_psnr)}')
        print(f'the mean psnr reg is {np.mean(reg_psnr)}')
        print(f'the mean ssim forward is {np.mean(forward_ssim)}')
        print(f'the mean ssim reg is {np.mean(reg_ssim)}')
        print('--------------------------------')
        if np.mean(forward_psnr) < np.mean(reg_psnr):
            self.regression.save_weights('model_weight/best_regression')

        total_psnr = list(zip(forward_psnr, reg_psnr))
        total_ssim = list(zip(forward_ssim, reg_ssim))
        for num, (forward, reg) in enumerate(total_psnr):
            plt.plot(forward, marker='.')
            plt.plot(reg, marker='.')

            plt.legend([f'mPSNR={str(np.mean(forward))[0:5]}', f'mPSNR={str(np.mean(reg))[0:5]}'], loc='upper right')
            plt.savefig(f'result/regression/AR_fine_tune/{epoch}-{2**(num + 1)}-ratio-PSNR')
            plt.close()

        for num, (forward, reg) in enumerate(total_ssim):
            plt.plot(forward, marker='.')
            plt.plot(reg, marker='.')
            plt.legend([f'mSSIM={str(np.mean(forward))[0:5]}', f'mSSIM={str(np.mean(reg))[0:5]}'], loc='upper right')
            plt.savefig(f'result/regression/AR_fine_tune/{epoch}-{2**(num+1)}-ratio-SSIM')
            plt.close()

    def main(self):
        image_loss_epoch = []
        style_loss_epoch = []
        reg_loss_epoch = []
        adv_loss_epoch = []
        constrast_loss_epoch = []


        # random shuffle data
        data = list(zip(self.zdH, self.zdh, self.zdm, self.zdl, self.zgH, self.zgh, self.zgm, self.zgl, self.train_path))
        np.random.shuffle(data)
        data = list(zip(*data))
        zdH, zdh, zdm, zdl, zgH, zgh, zgm, zgl, train_path = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), np.array(data[4]), np.array(data[5]), np.array(data[6]), np.array(data[7]), np.array(data[8])
        opti = tf.keras.optimizers.Adam(2e-4)


        for epoch in range(1, self.epochs + 1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            reg_loss_batch = []
            adv_loss_batch = []
            constrast_loss_batch = []

            for batch in range(self.batch_num):
                batch_high_images = self.get_batch_data(train_path, batch, self.batch_size, image=True)
                batch_zdH = self.get_batch_data(zdH, batch, self.batch_size)
                batch_zdh = self.get_batch_data(zdh, batch, self.batch_size)
                batch_zdm = self.get_batch_data(zdm, batch, self.batch_size)
                batch_zdl = self.get_batch_data(zdl, batch, self.batch_size)
                batch_zgH = self.get_batch_data(zgH, batch, self.batch_size)
                batch_zgh = self.get_batch_data(zgh, batch, self.batch_size)
                batch_zgm = self.get_batch_data(zgm, batch, self.batch_size)
                batch_zgl = self.get_batch_data(zgl, batch, self.batch_size)


                reg_loss, image_loss, style_loss, adv_loss, constrast_loss = \
                self.train_step(batch_high_images, batch_zdH, batch_zdh, batch_zdm, batch_zdl, batch_zgH, batch_zgh, batch_zgm, batch_zgl, opti)

                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                reg_loss_batch.append(reg_loss)
                adv_loss_batch.append(adv_loss)
                constrast_loss_batch.append(constrast_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            reg_loss_epoch.append(np.mean(reg_loss_batch))
            adv_loss_epoch.append(np.mean(adv_loss_batch))
            constrast_loss_epoch.append(np.mean(constrast_loss_batch))
            print('------------')
            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the reg_loss is {reg_loss_epoch[-1]}')
            print(f'the adv_loss is {adv_loss_epoch[-1]}')
            print(f'the constrast_loss is {constrast_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')
            self.regression.save_weights('model_weight/regression_one_to_one3')
            self.plot_image_AR(epoch, train=False)
            filename = 'AR_fine_tune'

            plt.plot(image_loss_epoch)
            plt.savefig(f'result/regression/{filename}/image_loss')
            plt.close()

            plt.plot(style_loss_epoch)
            plt.savefig(f'result/regression/{filename}/style_loss')
            plt.close()

            plt.plot(reg_loss_epoch)
            plt.savefig(f'result/regression/{filename}/reg_loss')
            plt.close()

            plt.plot(adv_loss_epoch)
            plt.savefig(f'result/regression/{filename}/adv_loss')
            plt.close()

            plt.plot(constrast_loss_epoch)
            plt.savefig(f'result/regression/{filename}/constrast_loss')
            plt.close()


def pre_train_regression():
    global encoder
    global ztozd
    global ztozg
    global regression
    global generator
    encoder = normal_encoder()
    ztozd = ZtoZd()
    ztozg = ZtoZg()
    regression = regression_model_with_instance()
    generator = generator()
    encoder.load_weights('model_weight/AE_encoder')
    ztozd.load_weights('model_weight/AE_ztozd')
    ztozg.load_weights('model_weight/zd_zg_distillation_ztozg')
    generator.load_weights('model_weight/zd_zg_distillation_generator')

    def prepare_data():
        path_AR_syn_train = '/home/bosen/PycharmProjects/Datasets/AR_train/'
        path_AR_syn_test = '/home/bosen/PycharmProjects/Datasets/AR_test/'

        train_database_feature_zg, train_database_feature_zd = [], []
        test_database_feature_zg, test_database_feature_zd = [], []
        ID = [f'ID{i}' for i in range(1, 91)]
        for num, id in enumerate(ID):
            for count, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_train + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    zH = encoder(image.reshape(1, 64, 64, 1))
                    zdH, _ = ztozd(zH)
                    zgH, _, _ =ztozg(zH)
                    train_database_feature_zd.append(tf.reshape(zdH, [200]))
                    train_database_feature_zg.append(tf.reshape(zgH, [200]))

        ID = [f'ID0{i}' if i < 10 else f'ID{i}' for i in range(1, 22)]
        for num, id in enumerate(ID):
            for count, filename in enumerate(os.listdir(path_AR_syn_test + id)):
                if count < 20:
                    image = cv2.imread(path_AR_syn_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    zH = encoder(image.reshape(1, 64, 64, 1))
                    zdH, _ = ztozd(zH)
                    zgH, _, _ =ztozg(zH)
                    test_database_feature_zd.append(tf.reshape(zdH, [200]))
                    test_database_feature_zg.append(tf.reshape(zgH, [200]))

        return np.array(train_database_feature_zd), np.array(train_database_feature_zg), np.array(test_database_feature_zd), np.array(test_database_feature_zg)

    train_data_zd, train_data_zg, test_data_zd, test_data_zg = prepare_data()
    print(train_data_zd.shape, train_data_zg.shape, test_data_zd.shape, test_data_zg.shape)


    regression.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mse'])
    history = regression.fit((train_data_zg, train_data_zd), train_data_zg, epochs=80, batch_size=60, validation_data=([test_data_zg, test_data_zd], test_data_zg), verbose=1)
    regression.save_weights('model_weight/regression_pre_train')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    # path = 'result/regression/AR_fine_tune/'
    # for filename in os.listdir(path):
    #     os.remove(path + filename)
    # pre_train_regression()
    # reg = Regression(epochs=15, batch_size=10, batch_num=180)
    # reg.main()



