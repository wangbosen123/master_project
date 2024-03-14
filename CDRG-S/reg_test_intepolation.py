from experiment import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# def reg_overall_test(train=False):
#     global encoder
#     global reg
#     global generator
#     global discriminator
#
#     encoder = encoder()
#     reg = regression()
#     generator = generator()
#     discriminator = discriminator()
#
#     encoder.load_weights('weights/encoder')
#     reg.load_weights('weights/reg')
#     generator.load_weights('weights/generator2')
#     discriminator.load_weights('weights/discriminator2')
#
#     def down_image(image, ratio):
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
#     def style(real, fake):
#         feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#         real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
#         real = tf.image.grayscale_to_rgb(real)
#         fake = tf.image.grayscale_to_rgb(fake)
#
#         real_feature = feature_extraction(real)
#         fake_feature = feature_extraction(fake)
#         distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
#         return distance
#
#     def distillation_loss(target_z, target_zreg, database_z, database_zreg, mean_z, mean_zreg):
#         dis_loss = 0
#         for num, (latent_z, latent_zreg) in enumerate(zip(target_z, target_zreg)):
#             latent_z_expand = tf.tile(tf.reshape(latent_z, [-1, 200]), [database_z.shape[0], 1])
#
#             z_neighbor_distance = tf.sqrt(tf.reduce_sum(tf.square(latent_z_expand - database_z), axis=-1))
#             z_neighbor_distance = z_neighbor_distance.numpy()
#
#             for i in range(3):
#                 min_index = np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]
#                 dis_loss += tf.math.abs(((tf.sqrt(tf.reduce_sum(tf.square(latent_z - database_z[min_index])))/ mean_z) - (tf.sqrt(tf.reduce_sum(tf.square(latent_zreg - database_zreg[min_index])))) / mean_zreg))
#                 z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
#
#         return dis_loss / (num + 1)
#
#     def get_database():
#         train_path = '/disk2/bosen/Datasets/AR_train/'
#         test_path = '/disk2/bosen/Datasets/AR_test/'
#
#         database_z, database_zreg = [], []
#         for id_num, id in enumerate(os.listdir(train_path)):
#             for file_num, filename in enumerate(os.listdir(train_path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(train_path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     z = encoder(image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     database_z.append(tf.reshape(z, [200]))
#                     database_zreg.append(tf.reshape(zreg, [200]))
#
#         for id_num, id in enumerate(os.listdir(test_path)):
#             for file_num, filename in enumerate(os.listdir(test_path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(test_path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     z = encoder(image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     database_z.append(tf.reshape(z, [200]))
#                     database_zreg.append(tf.reshape(zreg, [200]))
#
#         database_z, database_zreg = np.array(database_z), np.array(database_zreg)
#
#         dot_product_z_space = tf.matmul(database_z, tf.transpose(database_z))
#         dot_product_zreg_space = tf.matmul(database_zreg, tf.transpose(database_zreg))
#         square_norm_z_space = tf.linalg.diag_part(dot_product_z_space)
#         square_norm_zreg_space = tf.linalg.diag_part(dot_product_zreg_space)
#
#         distances_z = tf.sqrt(tf.expand_dims(square_norm_z_space, 1) - 2.0 * dot_product_z_space + tf.expand_dims(square_norm_z_space, 0) + 1e-8) / 2
#         distances_zreg = tf.sqrt(tf.expand_dims(square_norm_zreg_space, 1) - 2.0 * dot_product_zreg_space + tf.expand_dims(square_norm_zreg_space, 0) + 1e-8) / 2
#
#         mean_distances_z = (tf.reduce_sum(distances_z)) / (math.factorial(distances_z.shape[0]) / (math.factorial(2) * math.factorial(int(distances_z.shape[0] - 2))))
#         mean_distances_zreg = (tf.reduce_sum(distances_zreg)) / (math.factorial(distances_zreg.shape[0]) / (math.factorial(2) * math.factorial(int(distances_zreg.shape[0] - 2))))
#
#         return database_z, database_zreg, mean_distances_z, mean_distances_zreg
#
#     if train: path = '/disk2/bosen/Datasets/AR_train/'
#     else: path = '/disk2/bosen/Datasets/AR_test/'
#
#     def search(ratio):
#         if ratio == 2:
#             img_min, img_max = 0.00302, 0.00451
#             sty_min, sty_max = 0.0307, 0.0462
#             adv_min, adv_max = 0.298, 0.304
#             dis_min, dis_max = 0.238, 0.670
#         elif ratio == 4:
#             img_min, img_max = 0.00297, 0.00551
#             sty_min, sty_max = 0.015, 0.023
#             adv_min, adv_max = 0.298, 0.313
#             dis_min, dis_max = 0.180, 0.984
#         elif ratio == 8:
#             img_min, img_max = 0.0032, 0.1036
#             sty_min, sty_max = 0.0086, 0.011
#             adv_min, adv_max = 0.299, 0.348
#             dis_min, dis_max = 0.089, 1.552
#
#         database_latent_z, database_latent_zreg, database_gt, database_image = [], [], [], []
#         for id_num, id in enumerate(os.listdir(path)):
#             for file_num, filename in enumerate(os.listdir(path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#
#                     if ratio == 1:
#                         low_image = image
#                     elif ratio == 2:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     elif ratio == 4:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                     elif ratio == 8:
#                         low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#
#                     zH = encoder(image.reshape(1, 64, 64, 1))
#                     z = encoder(low_image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     zH, z, zreg = tf.reshape(zH, [200]), tf.reshape(z, [200]), tf.reshape(zreg, [200])
#                     database_image.append(low_image)
#                     database_gt.append(zH)
#                     database_latent_z.append(z)
#                     database_latent_zreg.append(zreg)
#
#         database_latent_z, database_latent_zreg, database_gt, database_image = np.array(database_latent_z), np.array(database_latent_zreg), np.array(database_gt), np.array(database_image)
#         database_z, database_zreg, mean_z, mean_zreg = get_database()
#
#         total_loss = [[0 for i in range(13)] for i in range(3)]
#         image_loss = [[0 for i in range(13)] for i in range(3)]
#         style_loss = [[0 for i in range(13)] for i in range(3)]
#         adv_loss = [[0 for i in range(13)] for i in range(3)]
#         dis_loss = [[0 for i in range(13)] for i in range(3)]
#         latent_score = [[0 for i in range(13)] for i in range(3)]
#
#         learning_rate, lr = [], 1
#         for i in range(10):
#             lr *= 0.8
#             learning_rate.append(lr)
#         learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 ,0.2, 0.1, 0.01, 0]
#         print(learning_rate)
#
#         syn = generator(tf.reshape(database_latent_z, [-1, 200]))
#         down_syn = down_image(syn, ratio)
#         syn_score = discriminator(syn)
#
#         latent_score[0][0] = (tf.reduce_mean(tf.square(database_latent_z - database_gt)))
#         latent_score[1][0] = (tf.reduce_mean(tf.square(database_latent_z - database_gt)))
#         latent_score[2][0] = (tf.reduce_mean(tf.square(database_latent_z - database_gt)))
#
#         image_loss[0][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)
#         image_loss[1][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)
#         image_loss[2][0] = (tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)
#
#         style_loss[0][0] = (style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)
#         style_loss[1][0] = (style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)
#         style_loss[2][0] = (style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)
#
#         adv_loss[0][0] = (tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)
#         adv_loss[1][0] = (tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)
#         adv_loss[2][0] = (tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)
#
#         dis_loss[0][0] = ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min))
#         dis_loss[1][0] = ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min))
#         dis_loss[2][0] = ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min))
#
#         total_loss[0][0] = (((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                             ((style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)) +
#                             ((tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                             ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#
#         total_loss[1][0] = (((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                             ((style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)) +
#                             ((tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                             ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#
#         total_loss[2][0] = (((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                             ((style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)) +
#                             ((tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                             ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#
#         res_loss = (((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                     ((style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)) +
#                     ((tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                     ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#
#         print(res_loss)
#
#         zg = database_latent_z
#         z_final = zg
#         for t in range(3):
#             _, _, zreg = reg(zg)
#             dzreg = zreg - zg
#             for index, w in enumerate(learning_rate):
#                 z_output = zg + (w * dzreg)
#                 reg_syn = generator(z_output)
#                 reg_syn_score = discriminator(reg_syn)
#                 reg_down_syn = down_image(reg_syn, ratio)
#                 if (((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                         ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) +
#                         ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                         ((distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)) < res_loss):
#
#                     res_loss = (((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                                 ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) +
#                                 ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                                 ((distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#                     z_final = zg + (w * dzreg)
#
#
#                 total_loss[t][index + 1] = (((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                                             ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) +
#                                             ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                                             ((distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#
#                 image_loss[t][index + 1] = ((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min))
#                 style_loss[t][index + 1] = ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min))
#                 adv_loss[t][index + 1] = ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min))
#                 dis_loss[t][index + 1] = ((distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min))
#                 latent_score[t][index + 1] = (tf.reduce_mean(tf.square(z_output - database_gt)))
#             zg = z_final
#
#         for loss in [[total_loss, 'total_loss'], [latent_score, 'latent error'], [image_loss, 'image_loss'], [style_loss, 'style_loss'], [adv_loss, 'adv_loss'], [dis_loss, 'dis_loss']]:
#             loss_curve = np.array(loss[0])
#             loss_name = loss[1]
#             x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
#             plt.plot(x, loss_curve[0], marker='o')
#             plt.plot(x, loss_curve[1], marker='o')
#             plt.plot(x, loss_curve[2], marker='o')
#             plt.title(loss_name)
#             plt.xlabel('W')
#             plt.ylabel('Error')
#             plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')
#
#             for y in range(3):
#                 for i, j in zip(x, loss_curve[y]):
#                     plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
#             plt.show()
#
#         return total_loss, latent_score, image_loss, style_loss, adv_loss
#
#     _, _, image1_loss, style1_loss, adv1_loss = search(ratio = 2)
#     _, _, image2_loss, style2_loss, adv2_loss = search(ratio = 4)
#     _, _, image3_loss, style3_loss, adv3_loss = search(ratio = 8)
#
#     # print(image1_loss, np.min(image1_loss), np.max(image1_loss))
#     # print(style1_loss, np.min(style1_loss), np.max(style1_loss))
#     # print(adv1_loss, np.min(adv1_loss), np.max(adv1_loss))
#     #
#     # print(image2_loss, np.min(image2_loss), np.max(image2_loss))
#     # print(style2_loss, np.min(style2_loss), np.max(style2_loss))
#     # print(adv2_loss, np.min(adv2_loss), np.max(adv2_loss))
#     #
#     # print(image3_loss, np.min(image3_loss), np.max(image3_loss))
#     # print(style3_loss, np.min(style3_loss), np.max(style3_loss))
#     # print(adv3_loss, np.min(adv3_loss), np.max(adv3_loss))

# def reg_single_test(train=False):
#     global encoder
#     global reg
#     global generator
#     global discriminator
#
#     encoder = encoder()
#     reg = regression()
#     generator = generator()
#     discriminator = discriminator()
#
#     encoder.load_weights('weights/encoder')
#     reg.load_weights('weights/reg')
#     generator.load_weights('weights/generator2')
#     discriminator.load_weights('weights/discriminator2')
#
#     def down_image(image, ratio):
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
#     def style(real, fake):
#         feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
#         real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
#         real = tf.image.grayscale_to_rgb(real)
#         fake = tf.image.grayscale_to_rgb(fake)
#
#         real_feature = feature_extraction(real)
#         fake_feature = feature_extraction(fake)
#         distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
#         return distance
#
#     def distillation_loss(target_z, target_zreg, database_z, database_zreg, mean_z, mean_zreg):
#         dis_loss = 0
#         for num, (latent_z, latent_zreg) in enumerate(zip(target_z, target_zreg)):
#             latent_z_expand = tf.tile(tf.reshape(latent_z, [-1, 200]), [database_z.shape[0], 1])
#
#             z_neighbor_distance = tf.sqrt(tf.reduce_sum(tf.square(latent_z_expand - database_z), axis=-1))
#             z_neighbor_distance = z_neighbor_distance.numpy()
#
#             for i in range(3):
#                 min_index = np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]
#                 dis_loss += tf.math.abs(((tf.sqrt(tf.reduce_sum(tf.square(latent_z - database_z[min_index])))/ mean_z) - (tf.sqrt(tf.reduce_sum(tf.square(latent_zreg - database_zreg[min_index])))) / mean_zreg))
#                 z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
#
#         return dis_loss / (num + 1)
#
#     def get_database():
#         train_path = '/disk2/bosen/Datasets/AR_train/'
#         test_path = '/disk2/bosen/Datasets/AR_test/'
#
#         database_z, database_zreg = [], []
#         for id_num, id in enumerate(os.listdir(train_path)):
#             for file_num, filename in enumerate(os.listdir(train_path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(train_path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     z = encoder(image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     database_z.append(tf.reshape(z, [200]))
#                     database_zreg.append(tf.reshape(zreg, [200]))
#
#         for id_num, id in enumerate(os.listdir(test_path)):
#             for file_num, filename in enumerate(os.listdir(test_path + id)):
#                 if 0 <= file_num < 5:
#                     image = cv2.imread(test_path + id + '/' + filename, 0) / 255
#                     image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                     z = encoder(image.reshape(1, 64, 64, 1))
#                     _, _, zreg = reg(z)
#                     database_z.append(tf.reshape(z, [200]))
#                     database_zreg.append(tf.reshape(zreg, [200]))
#
#         database_z, database_zreg = np.array(database_z), np.array(database_zreg)
#
#         dot_product_z_space = tf.matmul(database_z, tf.transpose(database_z))
#         dot_product_zreg_space = tf.matmul(database_zreg, tf.transpose(database_zreg))
#         square_norm_z_space = tf.linalg.diag_part(dot_product_z_space)
#         square_norm_zreg_space = tf.linalg.diag_part(dot_product_zreg_space)
#
#         distances_z = tf.sqrt(tf.expand_dims(square_norm_z_space, 1) - 2.0 * dot_product_z_space + tf.expand_dims(square_norm_z_space, 0) + 1e-8) / 2
#         distances_zreg = tf.sqrt(tf.expand_dims(square_norm_zreg_space, 1) - 2.0 * dot_product_zreg_space + tf.expand_dims(square_norm_zreg_space, 0) + 1e-8) / 2
#
#         mean_distances_z = (tf.reduce_sum(distances_z)) / (math.factorial(distances_z.shape[0]) / (math.factorial(2) * math.factorial(int(distances_z.shape[0] - 2))))
#         mean_distances_zreg = (tf.reduce_sum(distances_zreg)) / (math.factorial(distances_zreg.shape[0]) / (math.factorial(2) * math.factorial(int(distances_zreg.shape[0] - 2))))
#
#         return database_z, database_zreg, mean_distances_z, mean_distances_zreg
#
#     if train: path = '/disk2/bosen/Datasets/AR_train/'
#     else: path = '/disk2/bosen/Datasets/AR_test/ID18/'
#
#     def search(ratio):
#         if ratio == 2:
#             img_min, img_max = 0.00302, 0.00451
#             sty_min, sty_max = 0.0307, 0.0462
#             adv_min, adv_max = 0.298, 0.304
#             dis_min, dis_max = 0.238, 0.670
#         elif ratio == 4:
#             img_min, img_max = 0.00297, 0.00551
#             sty_min, sty_max = 0.015, 0.023
#             adv_min, adv_max = 0.298, 0.313
#             dis_min, dis_max = 0.180, 0.984
#         elif ratio == 8:
#             img_min, img_max = 0.0032, 0.1036
#             sty_min, sty_max = 0.0086, 0.011
#             adv_min, adv_max = 0.299, 0.348
#             dis_min, dis_max = 0.089, 1.552
#
#         database_latent_z, database_latent_zreg, database_gt, database_image, database_high_image = [], [], [], [], []
#         for file_num, filename in enumerate(os.listdir(path)):
#             if 0 <= file_num < 1:
#                 image = cv2.imread(path + '/' + filename, 0) / 255
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
#                 zH = encoder(image.reshape(1, 64, 64, 1))
#                 z = encoder(low_image.reshape(1, 64, 64, 1))
#                 _, _, zreg = reg(z)
#                 zH, z, zreg = tf.reshape(zH, [200]), tf.reshape(z, [200]), tf.reshape(zreg, [200])
#                 database_high_image.append(image)
#                 database_image.append(low_image)
#                 database_gt.append(zH)
#                 database_latent_z.append(z)
#                 database_latent_zreg.append(zreg)
#
#         database_latent_z, database_latent_zreg, database_gt, database_image, database_high_image = np.array(database_latent_z), np.array(database_latent_zreg), np.array(database_gt), np.array(database_image), np.array(database_high_image)
#         database_z, database_zreg, mean_z, mean_zreg = get_database()
#
#         learning_rate, lr = [], 1
#         for i in range(10):
#             lr *= 0.8
#             learning_rate.append(lr)
#         learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 ,0.2, 0.1, 0.01, 0]
#
#         syn = generator(tf.reshape(database_latent_z, [-1, 200]))
#         down_syn = down_image(syn, ratio)
#         syn_score = discriminator(syn)
#         res_loss = (((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                     ((style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)) +
#                     ((tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                     ((distillation_loss(database_latent_zreg, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#
#
#         zg = database_latent_z
#         z_final = zg
#
#         for t in range(3):
#             _, _, zreg = reg(zg)
#             dzreg = zreg - zg
#             for index, w in enumerate(learning_rate):
#                 z_output = zg + (w * dzreg)
#                 reg_syn = generator(z_output)
#                 reg_syn_score = discriminator(reg_syn)
#                 reg_down_syn = down_image(reg_syn, ratio)
#                 if (((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                         ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) +
#                         ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                         ((distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)) < res_loss):
#
#                     res_loss = (((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) +
#                                 ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) +
#                                 ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min)) +
#                                 ((distillation_loss(z_output, database_latent_z, database_z, database_zreg, mean_z, mean_zreg) - dis_min) / (dis_max - dis_min)))
#                     z_final = zg + (w * dzreg)
#
#             zg = z_final
#             final_syn = generator(z_final)
#
#         plt.subplots(figsize=(5, 1))
#         plt.subplots_adjust(wspace=0, hspace=0)
#
#         plt.subplot(1, 5, 1)
#         plt.axis('off')
#         plt.imshow(database_high_image.reshape(64, 64), cmap='gray')
#
#         plt.subplot(1, 5, 2)
#         plt.axis('off')
#         plt.imshow(database_image.reshape(64, 64), cmap='gray')
#
#         plt.subplot(1, 5, 3)
#         plt.axis('off')
#         syn = generator(encoder(database_image.reshape(1, 64, 64, 1)))
#         print(encoder(database_image.reshape(1, 64, 64, 1)))
#         print('-----------------------------------')
#         plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')
#
#         plt.subplot(1, 5, 4)
#         plt.axis('off')
#         z = encoder(database_image.reshape(1, 64, 64, 1))
#         _, _, zreg = reg(z)
#         print(zreg)
#         print('-----------------------------------')
#         syn = generator(zreg)
#         plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')
#
#         plt.subplot(1, 5, 5)
#         plt.axis('off')
#         plt.imshow(tf.reshape(final_syn, [64, 64]), cmap='gray')
#         print(z_final)
#         plt.show()
#
#
#     search(2)
#     search(4)
#     search(8)



def reg_single_test(train=False):
    global encoder
    global reg
    global generator
    global discriminator

    encoder = encoder()
    reg = regression()
    generator = generator()
    discriminator = discriminator()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg')
    generator.load_weights('weights/generator2')
    discriminator.load_weights('weights/discriminator2')

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
    else: path = '/disk2/bosen/Datasets/AR_aligment_other/ID46/'
    # else: path = '/disk2/bosen/Datasets/AR_test/ID19/'

    def search(ratio):
        if ratio == 2:
            img_min, img_max = 0.00302, 0.00451
            sty_min, sty_max = 0.0307, 0.0462
            adv_min, adv_max = 0.298, 0.304
            dis_min, dis_max = 0.238, 0.670
        elif ratio == 4:
            img_min, img_max = 0.00297, 0.00551
            sty_min, sty_max = 0.015, 0.023
            adv_min, adv_max = 0.298, 0.313
            dis_min, dis_max = 0.180, 0.984
        elif ratio == 8:
            img_min, img_max = 0.0032, 0.1036
            sty_min, sty_max = 0.0086, 0.011
            adv_min, adv_max = 0.299, 0.348
            dis_min, dis_max = 0.089, 1.552

        database_latent_z, database_latent_zreg, database_gt, database_image, database_high_image = [], [], [], [], []
        for file_num, filename in enumerate(os.listdir(path)):
            if 0 <= file_num < 1:
                image = cv2.imread(path + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                if ratio == 1:
                    low_image = image
                elif ratio == 2:
                    low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 4:
                    low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 8:
                    low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                zH = encoder(image.reshape(1, 64, 64, 1))
                z = encoder(low_image.reshape(1, 64, 64, 1))
                _, _, zreg = reg(z)
                zH, z, zreg = tf.reshape(zH, [200]), tf.reshape(z, [200]), tf.reshape(zreg, [200])
                database_high_image.append(image)
                database_image.append(low_image)
                database_gt.append(zH)
                database_latent_z.append(z)
                database_latent_zreg.append(zreg)

        database_latent_z, database_latent_zreg, database_gt, database_image, database_high_image = np.array(database_latent_z), np.array(database_latent_zreg), np.array(database_gt), np.array(database_image), np.array(database_high_image)

        learning_rate, lr = [], 1
        for i in range(10):
            lr *= 0.8
            learning_rate.append(lr)
        learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 ,0.2, 0.1, 0.01, 0]

        syn = generator(tf.reshape(database_latent_z, [-1, 200]))
        down_syn = down_image(syn, ratio)
        # res_loss = ((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) + ((style(database_image.reshape(-1, 64, 64, 1), down_syn) - sty_min) / (sty_max - sty_min)) + ((tf.reduce_mean(tf.square(syn_score - 1)) - adv_min) / (adv_max - adv_min))
        res_loss = ((tf.reduce_mean(tf.square(down_syn - database_image.reshape(-1, 64, 64, 1)))))

        zg = database_latent_z
        z_final = zg
        for t in range(3):
            _, _, zreg = reg(zg)
            dzreg = zreg - zg
            for index, w in enumerate(learning_rate):
                z_output = zg + (w * dzreg)
                reg_syn = generator(z_output)
                reg_syn_score = discriminator(reg_syn)
                reg_down_syn = down_image(reg_syn, ratio)
                # if ((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) + ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) + ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min)) < res_loss:
                #     res_loss = ((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))) - img_min) / (img_max - img_min)) + ((style(database_image.reshape(-1, 64, 64, 1), reg_down_syn) - sty_min) / (sty_max - sty_min)) + ((tf.reduce_mean(tf.square(reg_syn_score - 1)) - adv_min) / (adv_max - adv_min))
                if ((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1))))) < res_loss:
                    res_loss = ((tf.reduce_mean(tf.square(reg_down_syn - database_image.reshape(-1, 64, 64, 1)))))
                    z_final = zg + (w * dzreg)

            zg = z_final
            final_syn = generator(z_final)

        plt.subplots(figsize=(5, 1))
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.subplot(1, 5, 1)
        plt.axis('off')
        plt.imshow(database_high_image.reshape(64, 64), cmap='gray')

        plt.subplot(1, 5, 2)
        plt.axis('off')
        plt.imshow(database_image.reshape(64, 64), cmap='gray')

        plt.subplot(1, 5, 3)
        plt.axis('off')
        syn = generator(encoder(database_image.reshape(1, 64, 64, 1)))
        plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        plt.subplot(1, 5, 4)
        plt.axis('off')
        z = encoder(database_image.reshape(1, 64, 64, 1))
        _, _, zreg = reg(z)
        syn = generator(zreg)
        plt.imshow(tf.reshape(syn, [64, 64]), cmap='gray')

        plt.subplot(1, 5, 5)
        plt.axis('off')
        plt.imshow(tf.reshape(final_syn, [64, 64]), cmap='gray')
        plt.show()

        print(tf.image.psnr(tf.cast(database_high_image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(generator(encoder(database_image.reshape(1, 64, 64, 1))), dtype=tf.float32), max_val=1)[0])
        print(tf.image.psnr(tf.cast(database_high_image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
        print(tf.image.psnr(tf.cast(database_high_image.reshape(1, 64, 64, 1), dtype=tf.float32), tf.cast(final_syn, dtype=tf.float32), max_val=1)[0])
        print('-----')


    search(8)
    search(4)
    search(2)


def reg_overall_test_cls(train=False):
    global encoder
    global reg
    global generator
    global cls
    global discriminator

    encoder = encoder()
    reg = regression()
    cls = cls()
    generator = generator()
    discriminator = discriminator()

    encoder.load_weights('weights/encoder')
    reg.load_weights('weights/reg_x_cls_REG')
    cls.load_weights('weights/reg_x_cls_CLS')
    generator.load_weights('weights/generator2')
    discriminator.load_weights('weights/discriminator2')

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

    def database():
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
                z1 = encoder(low1_image.reshape(1, 64, 64, 1))
                z2 = encoder(low2_image.reshape(1, 64, 64, 1))
                z3 = encoder(low3_image.reshape(1, 64, 64, 1))
                z_database.append(tf.reshape(z1, [200]))
                z_database.append(tf.reshape(z2, [200]))
                z_database.append(tf.reshape(z3, [200]))
                _, _, zreg1 = reg(z1)
                _, _, zreg2 = reg(z2)
                _, _, zreg3 = reg(z3)
                zreg_database.append(tf.reshape(zreg1, [200]))
                zreg_database.append(tf.reshape(zreg2, [200]))
                zreg_database.append(tf.reshape(zreg3, [200]))
        z_database, zreg_database = np.array(z_database), np.array(zreg_database)
        return z_database, zreg_database

    # def distillation_loss(target_z, target_zreg, database_z, database_zreg):
    #     neighbor_z_error, neighbor_zreg_error = [], []
    #     latent_z_expand = tf.tile(tf.reshape(target_z, [-1, 200]), [database_z.shape[0], 1])
    #     z_neighbor_distance = tf.reduce_sum(tf.square(latent_z_expand - database_z), axis=-1)
    #     z_neighbor_distance = z_neighbor_distance.numpy()
    #     for i in range(3):
    #         min_index = np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]
    #         neighbor_z_error.append(tf.reduce_sum(tf.square(target_z - database_z[min_index])))
    #         neighbor_zreg_error.append(tf.reduce_sum(tf.square(target_zreg - database_zreg[min_index])))
    #         z_neighbor_distance[np.where(z_neighbor_distance == np.min(z_neighbor_distance))[0][0]] = np.max(z_neighbor_distance)
    #
    #     neighbor_z_error, neighbor_zreg_error = np.array(neighbor_z_error), np.array(neighbor_zreg_error)
    #     dist_loss = sum(abs((neighbor_z_error / np.sum(neighbor_z_error)) - (neighbor_zreg_error / np.sum(neighbor_zreg_error))))
    #     return dist_loss

    def distillation_loss(target_z, target_zreg, database_z, database_zreg):
        target_z_expand = tf.tile(tf.reshape(target_z, [-1, 200]), [database_z.shape[0], 1])
        target_zreg_expand = tf.tile(tf.reshape(target_zreg, [-1, 200]), [database_zreg.shape[0], 1])

        distance_target_z_database_z = tf.reduce_sum(tf.square(database_z - target_z_expand), axis=-1)
        distance_target_zreg_database_zreg = tf.reduce_sum(tf.square(database_zreg - target_zreg_expand), axis=-1)

        sum_distance_z = tf.reduce_sum(distance_target_z_database_z)
        sum_distance_zreg = tf.reduce_sum(distance_target_zreg_database_zreg)

        dis_loss = abs((distance_target_z_database_z / sum_distance_z) - (distance_target_zreg_database_zreg / sum_distance_zreg))
        dis_loss = tf.reduce_sum(dis_loss)
        return dis_loss


    if train: path = '/disk2/bosen/Datasets/AR_train/'
    else: path = '/disk2/bosen/Datasets/AR_aligment_other/'
    # else: path = '/disk2/bosen/Datasets/AR_test/'

    def search(ratio):
        z_database, zreg_database = database()
        database_latent_z, database_high_image, database_low_image, database_id = [], [], [], []

        for id in (os.listdir(path)):
            for file_num, filename in enumerate(os.listdir(path + id)):
                if file_num == 1:
                    break
                image = cv2.imread(path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                if ratio == 1:
                    low_image = image
                elif ratio == 2:
                    low_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 4:
                    low_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                elif ratio == 8:
                    low_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                z = encoder(low_image.reshape(1, 64, 64, 1))
                z = tf.reshape(z, [200])
                database_high_image.append(image)
                database_low_image.append(low_image)
                database_latent_z.append(z)
                database_id.append(int(id[2:]) - 1)
        database_latent_z, database_high_image, database_low_image, database_id = np.array(database_latent_z), np.array(database_high_image), np.array(database_low_image), np.array(database_id)

        learning_rate, lr = [], 1
        for i in range(10):
            lr *= 0.8
            learning_rate.append(lr)
        learning_rate = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 ,0.2, 0.1, 0.01, 0]

        preds = []
        total_loss, image_loss, dis_loss = [[[] for _ in range(database_latent_z.shape[0])] for _ in range(3)], [[[] for _ in range(database_latent_z.shape[0])] for _ in range(3)], [[[] for _ in range(database_latent_z.shape[0])] for _ in range(3)]
        for num, (z, high_image, low_image, id) in enumerate(zip(database_latent_z, database_high_image, database_low_image, database_id)):
            print(num)
            z_init = z
            _, _, zreg = reg(tf.reshape(z, [1, 200]))
            syn = generator(tf.reshape(z, [-1, 200]))
            down_syn = down_image(syn, ratio)

            res_image_loss = 10 * ((tf.reduce_mean(tf.square(down_syn - low_image.reshape(-1, 64, 64, 1)))))
            res_dis_loss = distillation_loss(z, zreg, z_database, zreg_database)
            total_loss[0][num].append(res_image_loss +res_dis_loss)
            total_loss[1][num].append(res_image_loss +res_dis_loss)
            total_loss[2][num].append(res_image_loss +res_dis_loss)

            image_loss[0][num].append(res_image_loss)
            image_loss[1][num].append(res_image_loss)
            image_loss[2][num].append(res_image_loss)

            dis_loss[0][num].append(res_dis_loss)
            dis_loss[1][num].append(res_dis_loss)
            dis_loss[2][num].append(res_dis_loss)

            z = tf.reshape(z, [1, 200])
            z_final = z
            for t in range(3):
                _, _, zreg = reg(z)
                dzreg = zreg - z

                for index, w in enumerate(learning_rate):
                    z_output = z+ (w * dzreg)
                    reg_syn = generator(z_output)
                    reg_down_syn = down_image(reg_syn, ratio)
                    if 10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1))))) + distillation_loss(z, z_output, z_database, zreg_database) < (res_image_loss+res_dis_loss):
                    # if ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1))))) < (res_image_loss):
                        res_image_loss = 10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1)))))
                        res_dis_loss = distillation_loss(z, z_output, z_database, zreg_database)
                        z_final = z + (w * dzreg)
                    total_loss[t][num].append(10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1)))))  +  distillation_loss(z, z_output, z_database, zreg_database))
                    image_loss[t][num].append(10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1))))))
                    dis_loss[t][num].append(distillation_loss(z, z_output, z_database, zreg_database))

                z = z_final
            _, pred = cls(z)
            preds.append(np.argmax(pred, axis=-1)[0])

            before_reg_syn = generator(tf.reshape(z_init, [1, 200]))
            _, _, zreg_init = reg(tf.reshape(z_init, [1, 200]))
            after_reg_syn = generator(zreg_init)
            after_reg_opti_syn = generator(z_final)
            print(f'the before reg syn PSNR is {tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(before_reg_syn, dtype=tf.float32), max_val=1)[0]}')
            print(f'the after reg syn PSNR is {tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(after_reg_syn, dtype=tf.float32), max_val=1)[0]}')
            print(f'the after reg opti syn PSNR is {tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(after_reg_opti_syn, dtype=tf.float32), max_val=1)[0]}')

            plt.subplots(figsize=(5, 1))
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(1, 5, 1)
            plt.axis('off')
            plt.imshow(tf.reshape(high_image, [64, 64]), cmap='gray')
            plt.subplot(1, 5, 2)
            plt.axis('off')
            plt.imshow(tf.reshape(low_image, [64, 64]), cmap='gray')
            plt.subplot(1, 5, 3)
            plt.axis('off')
            plt.imshow(tf.reshape(before_reg_syn, [64, 64]), cmap='gray')
            plt.subplot(1, 5, 4)
            plt.axis('off')
            plt.imshow(tf.reshape(after_reg_syn, [64, 64]), cmap='gray')
            plt.subplot(1, 5, 5)
            plt.axis('off')
            plt.imshow(tf.reshape(after_reg_opti_syn, [64, 64]), cmap='gray')
            plt.show()

        preds = np.array(preds)
        accuracy = accuracy_score(database_id, preds)
        print(f'{ratio} ratio accuracy is {accuracy}')

        # plt.plot(tf.reduce_mean(image_loss[0], axis=0), marker='o')
        # plt.plot(tf.reduce_mean(image_loss[1], axis=0), marker='o')
        # plt.plot(tf.reduce_mean(image_loss[2], axis=0), marker='o')
        # plt.title('image_loss')
        # plt.legend(['reg1', 'reg2', 'reg3'], loc='upper right')
        # plt.show()
        #
        # plt.plot(tf.reduce_mean(dis_loss[0], axis=0), marker='o')
        # plt.plot(tf.reduce_mean(dis_loss[1], axis=0), marker='o')
        # plt.plot(tf.reduce_mean(dis_loss[2], axis=0), marker='o')
        # plt.title('image_loss')
        # plt.legend(['reg1', 'reg2', 'reg3'], loc='upper right')
        # plt.show()
        total_loss = [tf.reduce_mean(total_loss[0], axis=0), tf.reduce_mean(total_loss[1], axis=0), tf.reduce_mean(total_loss[2], axis=0)]
        rec_loss = [tf.reduce_mean(image_loss[0], axis=0), tf.reduce_mean(image_loss[1], axis=0), tf.reduce_mean(image_loss[2], axis=0)]
        distill_loss = [tf.reduce_mean(dis_loss[0], axis=0), tf.reduce_mean(dis_loss[1], axis=0), tf.reduce_mean(dis_loss[2], axis=0)]


        for loss in [[total_loss, 'total loss'], [rec_loss, 'rec loss'], [distill_loss, 'distillation loss']]:
            loss_curve = np.array(loss[0])
            loss_name = loss[1]
            x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

            plt.plot(x, loss_curve[0], marker='o')
            plt.plot(x, loss_curve[1], marker='o')
            plt.plot(x, loss_curve[2], marker='o')
            plt.title(f'{ratio} ratio' + loss_name)
            plt.xlabel('W')
            plt.ylabel('Error')
            plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')

            for y in range(3):
                for i, j in zip(x, loss_curve[y]):
                    plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
            plt.show()

    search(2)
    # search(4)
    # search(8)


# def reg_overall_test_cls():
#     global encoder
#     global reg
#     global generator
#     global cls
#     global discriminator
#
#     encoder = encoder()
#     reg = regression()
#     cls = cls()
#     generator = generator()
#
#     encoder.load_weights('weights/encoder')
#     reg.load_weights('weights/reg_x_cls_REG')
#     cls.load_weights('weights/reg_x_cls_CLS')
#     generator.load_weights('weights/generator2')
#
#     def down_image(image, ratio):
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
#         elif ratio == 5.3:
#             down_syn = tf.image.resize(image, [12, 12], method='bicubic')
#             down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
#         elif ratio == 8:
#             down_syn = tf.image.resize(image, [8, 8], method='bicubic')
#             down_syn = tf.image.resize(down_syn, [64, 64], method='bicubic')
#             return down_syn
#
#     def database():
#         z_database, zreg_database = [], []
#         path = '/disk2/bosen/Datasets/AR_train/'
#         for id in os.listdir(path):
#             for num, filename in enumerate(os.listdir(path + id)):
#                 if num == 2:
#                     break
#                 image = cv2.imread(path + id + '/' + filename, 0) / 255
#                 image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
#                 blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
#                 low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
#                 z1 = encoder(low1_image.reshape(1, 64, 64, 1))
#                 z2 = encoder(low2_image.reshape(1, 64, 64, 1))
#                 z3 = encoder(low3_image.reshape(1, 64, 64, 1))
#                 z_database.append(tf.reshape(z1, [200]))
#                 z_database.append(tf.reshape(z2, [200]))
#                 z_database.append(tf.reshape(z3, [200]))
#                 _, _, zreg1 = reg(z1)
#                 _, _, zreg2 = reg(z2)
#                 _, _, zreg3 = reg(z3)
#                 zreg_database.append(tf.reshape(zreg1, [200]))
#                 zreg_database.append(tf.reshape(zreg2, [200]))
#                 zreg_database.append(tf.reshape(zreg3, [200]))
#         z_database, zreg_database = np.array(z_database), np.array(zreg_database)
#         return z_database, zreg_database
#
#     def distillation_loss(target_z, target_zreg, database_z, database_zreg):
#         target_z_expand = tf.tile(tf.reshape(target_z, [-1, 200]), [database_z.shape[0], 1])
#         target_zreg_expand = tf.tile(tf.reshape(target_zreg, [-1, 200]), [database_zreg.shape[0], 1])
#
#         distance_target_z_database_z = tf.reduce_sum(tf.square(database_z - target_z_expand), axis=-1)
#         distance_target_zreg_database_zreg = tf.reduce_sum(tf.square(database_zreg - target_zreg_expand), axis=-1)
#
#         sum_distance_z = tf.reduce_sum(distance_target_z_database_z)
#         sum_distance_zreg = tf.reduce_sum(distance_target_zreg_database_zreg)
#
#         dis_loss = abs(
#             (distance_target_z_database_z / sum_distance_z) - (distance_target_zreg_database_zreg / sum_distance_zreg))
#         dis_loss = tf.reduce_sum(dis_loss)
#         return dis_loss
#
#     def rec_loss(syn, gt):
#         return 10 * tf.reduce_mean(tf.square(tf.reshape(gt, [1, 64, 64, 1]) - tf.reshape(syn, [1, 64, 64, 1])))
#
#     # path = '/disk2/bosen/Datasets/AR_aligment_other/'
#     path = '/disk2/bosen/Datasets/AR_test/'
#
#     def search(ratio):
#         z_database, zreg_database = database()
#         database_latent_z, database_high_image, database_low_image, database_id = [], [], [], []
#
#         for id in (os.listdir(path)):
#             for file_num, filename in enumerate(os.listdir(path + id)):
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
#                 z = encoder(low_image.reshape(1, 64, 64, 1))
#                 z = tf.reshape(z, [200])
#                 database_high_image.append(image)
#                 database_low_image.append(low_image)
#                 database_latent_z.append(z)
#                 database_id.append(int(id[2:]) - 1)
#         database_latent_z, database_high_image, database_low_image, database_id = np.array(database_latent_z), np.array(database_high_image), np.array(database_low_image), np.array(database_id)
#
#         learning_rate, lr = [], 1
#         for i in range(10):
#             lr *= 0.8
#             learning_rate.append(lr)
#         learning_rate = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0]
#
#         total_loss, image_loss, dis_loss = [[[] for _ in range(database_latent_z.shape[0])] for _ in range(3)], [[[] for _ in range(database_latent_z.shape[0])] for _ in range(3)], [[[] for _ in range(database_latent_z.shape[0])] for _ in range(3)]
#         PSNR, SSIM = [[] for i in range(3)], [[] for i in range(3)]
#         for num, (z, high_image, low_image, id) in enumerate(zip(database_latent_z, database_high_image, database_low_image, database_id)):
#             print(num)
#             _, _, zreg = reg(tf.reshape(z, [1, 200]))
#             syn = generator(tf.reshape(z, [-1, 200]))
#             reg_syn = generator(tf.reshape(zreg, [-1, 200]))
#             down_syn = down_image(syn, ratio)
#             PSNR[0].append(tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
#             PSNR[1].append(tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(reg_syn, dtype=tf.float32), max_val=1)[0])
#             SSIM[0].append(tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn, dtype=tf.float32), max_val=1)[0])
#             SSIM[1].append(tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(reg_syn, dtype=tf.float32), max_val=1)[0])
#
#
#             res_image_loss = 10 * ((tf.reduce_mean(tf.square(down_syn - low_image.reshape(-1, 64, 64, 1)))))
#             res_dis_loss = distillation_loss(z, zreg, z_database, zreg_database)
#             total_loss[0][num].append(res_image_loss + res_dis_loss)
#             total_loss[1][num].append(res_image_loss + res_dis_loss)
#             total_loss[2][num].append(res_image_loss + res_dis_loss)
#
#             image_loss[0][num].append(res_image_loss)
#             image_loss[1][num].append(res_image_loss)
#             image_loss[2][num].append(res_image_loss)
#
#             dis_loss[0][num].append(res_dis_loss)
#             dis_loss[1][num].append(res_dis_loss)
#             dis_loss[2][num].append(res_dis_loss)
#
#             z = tf.reshape(z, [1, 200])
#             z_final = z
#             for t in range(3):
#                 _, _, zreg = reg(z)
#                 dzreg = zreg - z
#
#                 for index, w in enumerate(learning_rate):
#                     z_output = z + (w * dzreg)
#                     reg_syn = generator(z_output)
#                     reg_down_syn = down_image(reg_syn, ratio)
#                     if 10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1))))) + distillation_loss(z, z_output, z_database, zreg_database) < (res_image_loss + res_dis_loss):
#                         res_image_loss = 10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1)))))
#                         res_dis_loss = distillation_loss(z, z_output, z_database, zreg_database)
#                         z_final = z + (w * dzreg)
#                     total_loss[t][num].append(10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1))))) + distillation_loss(z, z_output, z_database, zreg_database))
#                     image_loss[t][num].append(10 * ((tf.reduce_mean(tf.square(reg_down_syn - low_image.reshape(-1, 64, 64, 1))))))
#                     dis_loss[t][num].append(distillation_loss(z, z_output, z_database, zreg_database))
#                 z = z_final
#             PSNR[2].append(tf.image.psnr(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(generator(z), dtype=tf.float32), max_val=1)[0])
#             SSIM[2].append(tf.image.ssim(tf.cast(tf.reshape(high_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(generator(z), dtype=tf.float32), max_val=1)[0])
#
#         print(tf.reduce_mean(PSNR[0]))
#         print(tf.reduce_mean(PSNR[1]))
#         print(tf.reduce_mean(PSNR[2]))
#         print(tf.reduce_mean(SSIM[0]))
#         print(tf.reduce_mean(SSIM[1]))
#         print(tf.reduce_mean(SSIM[2]))
#
#
#         total_loss = [tf.reduce_mean(total_loss[0], axis=0), tf.reduce_mean(total_loss[1], axis=0), tf.reduce_mean(total_loss[2], axis=0)]
#         rec_loss = [tf.reduce_mean(image_loss[0], axis=0), tf.reduce_mean(image_loss[1], axis=0), tf.reduce_mean(image_loss[2], axis=0)]
#         distill_loss = [tf.reduce_mean(dis_loss[0], axis=0), tf.reduce_mean(dis_loss[1], axis=0), tf.reduce_mean(dis_loss[2], axis=0)]
#
#         for loss in [[total_loss, 'total loss'], [rec_loss, 'rec loss'], [distill_loss, 'distillation loss']]:
#             loss_curve = np.array(loss[0])
#             loss_name = loss[1]
#             x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#
#             plt.plot(x, loss_curve[0], marker='o')
#             plt.plot(x, loss_curve[1], marker='o')
#             plt.plot(x, loss_curve[2], marker='o')
#             plt.title(f'{ratio} ratio' + loss_name)
#             plt.xlabel('W')
#             plt.ylabel('Error')
#             plt.legend(['Search number=1', 'Search number=2', 'Search number=3'], loc='upper right')
#
#             for y in range(3):
#                 for i, j in zip(x, loss_curve[y]):
#                     plt.annotate(str(j)[0: 6], xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')
#             plt.show()
#
#     search(2)
#     # search(4)
#     # search(8)


if __name__ == '__main__':
    reg_overall_test_cls()


