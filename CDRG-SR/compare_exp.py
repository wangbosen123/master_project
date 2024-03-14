from experiment import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# def Unet():
#     ## unet网络结构下采样部分
#     # 输入层 第一部分
#     inputs = tf.keras.layers.Input(shape=(64, 64, 1))
#
#     x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.BatchNormalization()(x)  # 64*64*16
#     # 下采样
#     x1 = tf.keras.layers.MaxPooling2D(padding="same")(x)  # 32*32*16
#
#     # 卷积 第二部分
#     x1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x1)
#     x1 = tf.keras.layers.BatchNormalization()(x1)
#     x1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x1)
#     x1 = tf.keras.layers.BatchNormalization()(x1)  # 32*32*32
#     # 下采样
#     x2 = tf.keras.layers.MaxPooling2D(padding="same")(x1)  # 16*16*32
#
#     # 卷积 第三部分
#     x2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x2)
#     x2 = tf.keras.layers.BatchNormalization()(x2)
#     x2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x2)
#     x2 = tf.keras.layers.BatchNormalization()(x2)  # 16*16*64
#     # 下采样
#     x3 = tf.keras.layers.MaxPooling2D(padding="same")(x2)  # 8*8*64
#
#     # 卷积 第四部分
#     x3 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x3)
#     x3 = tf.keras.layers.BatchNormalization()(x3)
#     x3 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x3)
#     x3 = tf.keras.layers.BatchNormalization()(x3)  # 4*4*128
#     # 下采样
#     x4 = tf.keras.layers.MaxPooling2D(padding="same")(x3)  # 2*2*128
#     # 卷积  第五部分
#     x4 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x4)
#     x4 = tf.keras.layers.BatchNormalization()(x4)
#     x4 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x4)
#     x4 = tf.keras.layers.BatchNormalization()(x4)  # 2*2*256
#
#     ## unet网络结构上采样部分
#
#     # 反卷积 第一部分      512个卷积核 卷积核大小2*2 跨度2 填充方式same 激活relu
#     x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
#                                          padding="same",
#                                          activation="relu")(x4)  # 32*32*512
#     x5 = tf.keras.layers.BatchNormalization()(x5)
#     x6 = tf.concat([x3, x5], axis=-1)  # 合并 32*32*1024
#     # 卷积
#     x6 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x6)
#     x6 = tf.keras.layers.BatchNormalization()(x6)
#     x6 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x6)
#     x6 = tf.keras.layers.BatchNormalization()(x6)  # 32*32*512
#
#     # 反卷积 第二部分
#     x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,
#                                          padding="same",
#                                          activation="relu")(x6)  # 64*64*256
#     x7 = tf.keras.layers.BatchNormalization()(x7)
#     x8 = tf.concat([x2, x7], axis=-1)  # 合并 64*64*512
#     # 卷积
#     x8 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x8)
#     x8 = tf.keras.layers.BatchNormalization()(x8)
#     x8 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x8)
#     x8 = tf.keras.layers.BatchNormalization()(x8)  # #64*64*256
#
#     # 反卷积 第三部分
#     x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,
#                                          padding="same",
#                                          activation="relu")(x8)  # 128*128*128
#     x9 = tf.keras.layers.BatchNormalization()(x9)
#     x10 = tf.concat([x1, x9], axis=-1)  # 合并 128*128*256
#     # 卷积
#     x10 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x10)
#     x10 = tf.keras.layers.BatchNormalization()(x10)
#     x10 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x10)
#     x10 = tf.keras.layers.BatchNormalization()(x10)  # 128*128*128
#
#     # 反卷积 第四部分
#     x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,
#                                           padding="same",
#                                           activation="relu")(x10)  # 256*256*64
#     x11 = tf.keras.layers.BatchNormalization()(x11)
#     x12 = tf.concat([x, x11], axis=-1)  # 合并 256*256*128
#     # 卷积
#     x12 = tf.keras.layers.Conv2D(21, 3, padding="same", activation="relu")(x12)
#     x12 = tf.keras.layers.BatchNormalization()(x12)
#     x12 = tf.keras.layers.Conv2D(21, 3, padding="same", activation="relu")(x12)
#     x12 = tf.keras.layers.BatchNormalization()(x12)  # 256*256*64
#
#     # 输出层 第五部分
#     output = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x12)  # 256*256*34
#
#     return tf.keras.Model(inputs=inputs, outputs=output)


class PatchGAN():
    def __init__(self, epochs, batch_num, batch_size):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.g_opti = tf.keras.optimizers.Adam(1e-5)
        self.d_opti = tf.keras.optimizers.Adam(1e-5)

        #set the model
        self.encoder = encoder()
        self.reg = regression()
        self.generator = re_generator()
        self.g = generator()
        self.discriminator = discriminator()
        self.unet = unet()

        self.reg.load_weights('weights/reg_x_cls_REG')
        self.encoder.load_weights('weights/encoder')
        self.generator.load_weights('weights/unet_generator2')
        self.g.load_weights('weights/generator2')
        self.discriminator.load_weights('weights/discriminator2')
        self.unet.load_weights('weights/unet')
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

        #set the data path
        self.train_path, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        path_AR_syn_train = '/disk2/bosen/Datasets/AR_train/'
        path_AR_syn_test = '/disk2/bosen/Datasets/AR_test/'
        path_AR_real_train = "/disk2/bosen/Datasets/AR_original_alignment_train90/"
        train_path = []
        test_path1, test_path2, test_path3 = [], [], []
        ID = [f'ID{i}' for i in range(1, 91)]

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2070:
                train_path.append(path_celeba + filename)
            if num < 21:
                test_path3.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                if num == 21:
                    test_path2.append(path_AR_syn_train + id + '/' + filename)

        for count, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_AR_real_train + id)):
                if '-1-0' in filename or '-1-1' in filename or '-1-2' in filename:
                    train_path.append(path_AR_real_train + id + '/' + filename)

        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if num == 1:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)


        train_path, test_path1, test_path2, test_path3 = np.array(train_path), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        np.random.shuffle(train_path)
        return train_path, test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size):
        train_images, ground_truth = [], []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        for path in train_data:
            image = cv2.imread(path, 0) / 255
            if "AR" in path:
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            train_images.append(image), train_images.append(low1_image), train_images.append(low2_image), train_images.append(low3_image)
            ground_truth.append(image), ground_truth.append(image), ground_truth.append(image), ground_truth.append(image)

        ground_truth = np.array(ground_truth).reshape(-1, 64, 64, 1)
        train_images = np.array(train_images).reshape(-1, 64, 64, 1)
        return ground_truth, train_images

    def style_loss(self, real, fake):
        real, fake = tf.cast(real, dtype="float32"), tf.cast(fake, dtype="float32")
        real = tf.image.grayscale_to_rgb(real)
        fake = tf.image.grayscale_to_rgb(fake)

        real_feature = self.feature_extraction(real)
        fake_feature = self.feature_extraction(fake)
        distance = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return distance

    def g_train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z, out1, out2, out3, out4 = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_images = self.generator([zreg, out1, out2, out3, out4])
            fake_score = self.discriminator(gen_images)

            g_loss = tf.reduce_mean(tf.square(fake_score - 1))
            image_loss = 20 * tf.reduce_mean(tf.square(high_images - gen_images))
            style_loss = 5 * self.style_loss(high_images, gen_images)
            total_loss = g_loss + image_loss + style_loss

        if train:
            grads = tape.gradient(total_loss, self.generator.trainable_variables)
            self.g_opti.apply_gradients(zip(grads, self.generator.trainable_variables))
        return image_loss, style_loss, g_loss

    def d_train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z, out1, out2, out3, out4 = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_image = self.generator([zreg, out1, out2, out3, out4])
            real_score = self.discriminator(high_images)
            fake_score = self.discriminator(gen_image)
            d_loss = (tf.reduce_mean(tf.square(real_score - 1)) + tf.reduce_mean(tf.square(fake_score)))*0.5

        if train:
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opti.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return d_loss

    def training(self):
        image_loss_epoch = []
        style_loss_epoch = []
        g_loss_epoch = []
        d_loss_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            image_loss_batch = []
            style_loss_batch = []
            g_loss_batch = []
            d_loss_batch = []

            for batch in range(self.batch_num):
                high_images, low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                for i in range(2):
                    d_loss = self.d_train_step(low_images, high_images, train=True)
                d_loss_batch.append(d_loss)

                image_loss, style_loss, g_loss = self.g_train_step(low_images, high_images, train=True)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                g_loss_batch.append(g_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            g_loss_epoch.append(np.mean(g_loss_batch))
            d_loss_epoch.append(np.mean(d_loss_batch))
            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the style_loss is {style_loss_epoch[-1]}')
            print(f'the g_loss is {g_loss_epoch[-1]}')
            print(f'the d_loss is {d_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            # self.encoder.save_weights('weights/AE_GAN_E')
            self.generator.save_weights('weights/unet_generator2')
            # self.discriminator.save_weights('weights/AE_GAN_D')


            plt.plot(image_loss_epoch, label='Image Loss')
            plt.title('Image Loss')
            plt.savefig('result/GAN/unet_image_loss_AE_GAN')
            plt.close()

            plt.plot(style_loss_epoch, label='Style Loss')
            plt.title('Style Loss')
            plt.savefig('result/GAN/unet_style_loss_AE_GAN')
            plt.close()

            plt.plot(g_loss_epoch, label='G Loss')
            plt.title('G Loss')
            plt.savefig('result/GAN/unet_g_loss_AE_GAN')
            plt.close()

            plt.plot(d_loss_epoch, label='D Loss')
            plt.title('D Loss')
            plt.savefig('result/GAN/unet_d_loss_AE_GAN')
            plt.close()

            plt.plot(g_loss_epoch, label='G Loss')
            plt.plot(d_loss_epoch, label='D Loss')
            plt.legend(['G Loss', 'D Loss'], loc='upper right')
            plt.savefig('result/GAN/unet_adv_loss_AE_GAN')
            plt.close()

            path = '/disk2/bosen/Datasets/Test/'
            psnr, ssim = [[] for i in range(6)], [[] for i in range(6)]
            for file_num, filename in enumerate(os.listdir(path)):
                    image = cv2.imread(path + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                    low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low4_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low5_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    z, out1, out2, out3, out4 = self.encoder(image.reshape(1, 64, 64, 1))
                    syn0 = self.generator([z, out1, out2, out3, out4])

                    z, out1, out2, out3, out4 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                    syn1 = self.generator([z, out1, out2, out3, out4])

                    z, out1, out2, out3, out4 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                    syn2 = self.generator([z, out1, out2, out3, out4])

                    z, out1, out2, out3, out4 = self.encoder(low3_image.reshape(1, 64, 64, 1))
                    syn3 = self.generator([z, out1, out2, out3, out4])

                    z, out1, out2, out3, out4 = self.encoder(low4_image.reshape(1, 64, 64, 1))
                    syn4 = self.generator([z, out1, out2, out3, out4])

                    z, out1, out2, out3, out4 = self.encoder(low5_image.reshape(1, 64, 64, 1))
                    syn5 = self.generator([z, out1, out2, out3, out4])

                    psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                    psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                    psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                    psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                    psnr[4].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                    psnr[5].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

                    ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                    ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                    ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                    ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                    ssim[4].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                    ssim[5].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

            psnr, ssim = np.array(psnr), np.array(ssim)
            psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
            print(psnr)
            print(ssim)


    def test(self):
        path = '/disk2/bosen/Datasets/Test/'
        psnr, ssim = [[] for i in range(6)], [[] for i in range(6)]
        for file_num, filename in enumerate(os.listdir(path)):
            image = cv2.imread(path + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low4_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low5_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)



            z, out1, out2, out3, out4 = self.encoder(image.reshape(1, 64, 64, 1))
            # _, _, z = self.reg(z)
            syn0 = self.generator([z, out1, out2, out3, out4])

            z, out1, out2, out3, out4 = self.encoder(low1_image.reshape(1, 64, 64, 1))
            # _, _, z = self.reg(z)
            syn1 = self.generator([z, out1, out2, out3, out4])

            z, out1, out2, out3, out4 = self.encoder(low2_image.reshape(1, 64, 64, 1))
            # _, _, z = self.reg(z)
            syn2 = self.generator([z, out1, out2, out3, out4])

            z, out1, out2, out3, out4 = self.encoder(low3_image.reshape(1, 64, 64, 1))
            # _, _, z = self.reg(z)
            syn3 = self.generator([z, out1, out2, out3, out4])

            z, out1, out2, out3, out4 = self.encoder(low4_image.reshape(1, 64, 64, 1))
            # _, _, z =  self.reg(z)
            syn4 = self.generator([z, out1, out2, out3, out4])

            z, out1, out2, out3, out4 = self.encoder(low5_image.reshape(1, 64, 64, 1))
            # _, _, z = self.reg(z)
            syn5 = self.generator([z, out1, out2, out3, out4])

            # syn000 = self.unet(image.reshape(1, 64, 64, 1))
            # syn111 = self.unet(low1_image.reshape(1, 64, 64, 1))
            # syn222 = self.unet(low2_image.reshape(1, 64, 64, 1))
            # syn333 = self.unet(low3_image.reshape(1, 64, 64, 1))
            # syn444 = self.unet(low4_image.reshape(1, 64, 64, 1))
            # syn555 = self.unet(low5_image.reshape(1, 64, 64, 1))
            #
            # z0, _, _, _, _ = self.encoder(image.reshape(1, 64, 64, 1))
            # z1, _, _, _, _  = self.encoder(low1_image.reshape(1, 64, 64, 1))
            # z2, _, _, _, _  = self.encoder(low2_image.reshape(1, 64, 64, 1))
            # z3, _, _, _, _  = self.encoder(low3_image.reshape(1, 64, 64, 1))
            # z4, _, _, _, _  = self.encoder(low4_image.reshape(1, 64, 64, 1))
            # z5, _, _, _, _  = self.encoder(low5_image.reshape(1, 64, 64, 1))
            #
            # _, _, zreg0 = self.reg(z0)
            # _, _, zreg1 = self.reg(z1)
            # _, _, zreg2 = self.reg(z2)
            # _, _, zreg3 = self.reg(z3)
            # _, _, zreg4 = self.reg(z4)
            # _, _, zreg5 = self.reg(z5)
            #
            # syn00 = self.g(zreg0)
            # syn11 = self.g(zreg1)
            # syn22 = self.g(zreg2)
            # syn33 = self.g(zreg3)
            # syn44 = self.g(zreg4)
            # syn55 = self.g(zreg5)

            # plt.subplots(figsize=(4, 6))
            # plt.subplots_adjust(wspace=0, hspace=0)
            #
            # plt.subplot(6, 4, 1)
            # plt.axis('off')
            # plt.imshow(image, cmap='gray')
            # plt.subplot(6, 4, 5)
            # plt.axis('off')
            # low_image = cv2.resize(cv2.GaussianBlur(cv2.resize(image, (64, 64), cv2.INTER_CUBIC), (7, 7), 0), (32, 32), cv2.INTER_CUBIC)
            # plt.imshow(low_image, cmap='gray')
            # plt.subplot(6, 4, 9)
            # plt.axis('off')
            # low_image = cv2.resize(cv2.GaussianBlur(cv2.resize(image, (64, 64), cv2.INTER_CUBIC), (7, 7), 0), (20, 20), cv2.INTER_CUBIC)
            # plt.imshow(low_image, cmap='gray')
            # plt.subplot(6, 4, 13)
            # plt.axis('off')
            # low_image = cv2.resize(cv2.GaussianBlur(cv2.resize(image, (64, 64), cv2.INTER_CUBIC), (7, 7), 0), (16, 16), cv2.INTER_CUBIC)
            # plt.imshow(low_image, cmap='gray')
            # plt.subplot(6, 4, 17)
            # plt.axis('off')
            # low_image = cv2.resize(cv2.GaussianBlur(cv2.resize(image, (64, 64), cv2.INTER_CUBIC), (7, 7), 0), (10, 10), cv2.INTER_CUBIC)
            # plt.imshow(low_image, cmap='gray')
            # plt.subplot(6, 4, 21)
            # plt.axis('off')
            # low_image = cv2.resize(cv2.GaussianBlur(cv2.resize(image, (64, 64), cv2.INTER_CUBIC), (7, 7), 0), (8, 8), cv2.INTER_CUBIC)
            # plt.imshow(low_image, cmap='gray')
            #
            #
            # plt.subplot(6, 4, 2)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn0, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 6)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn1, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 10)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn2, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 14)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn3, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 18)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn4, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 22)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn5, [64, 64]), cmap='gray')
            #
            #
            # plt.subplot(6, 4, 3)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn00, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 7)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn11, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 11)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn22, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 15)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn33, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 19)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn44, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 23)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn55, [64, 64]), cmap='gray')
            #
            #
            # plt.subplot(6, 4, 4)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn000, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 8)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn111, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 12)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn222, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 16)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn333, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 20)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn444, [64, 64]), cmap='gray')
            # plt.subplot(6, 4, 24)
            # plt.axis('off')
            # plt.imshow(tf.reshape(syn555, [64, 64]), cmap='gray')
            # plt.show()


            psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
            psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
            psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
            psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
            psnr[4].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
            psnr[5].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

            ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
            ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
            ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
            ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
            ssim[4].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
            ssim[5].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

        psnr, ssim = np.array(psnr), np.array(ssim)
        psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
        print(psnr)
        print(ssim)


class Unet():
    def __init__(self, epochs, batch_num, batch_size):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.g_opti = tf.keras.optimizers.Adam(1e-4)

        #set the model
        self.encoder = encoder()
        self.reg = regression()
        # self.reg.load_weights('weights/reg_x_cls_REG')
        # self.encoder.load_weights('weights/encoder')
        self.generator = re_generator()

        #set the data path
        self.train_path, self.test_path1, self.test_path2, self.test_path3 = self.load_path()
        print(self.train_path.shape, self.test_path1.shape, self.test_path2.shape, self.test_path3.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        path_AR_syn_train = '/disk2/bosen/Datasets/AR_train/'
        path_AR_syn_test = '/disk2/bosen/Datasets/AR_test/'
        path_AR_real_train = "/disk2/bosen/Datasets/AR_original_alignment_train90/"
        train_path = []
        test_path1, test_path2, test_path3 = [], [], []
        ID = [f'ID{i}' for i in range(1, 91)]

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2070:
                train_path.append(path_celeba + filename)
            if num < 21:
                test_path3.append(path_celeba + filename)

        for id in ID:
            for num, filename in enumerate(os.listdir(path_AR_syn_train + id)):
                if num < 20:
                    train_path.append(path_AR_syn_train + id + '/' + filename)
                if num == 21:
                    test_path2.append(path_AR_syn_train + id + '/' + filename)

        for count, id in enumerate(ID):
            for num, filename in enumerate(os.listdir(path_AR_real_train + id)):
                if '-1-0' in filename or '-1-1' in filename or '-1-2' in filename:
                    train_path.append(path_AR_real_train + id + '/' + filename)

        for ID in os.listdir(path_AR_syn_test):
            for num, filename in enumerate(os.listdir(path_AR_syn_test + ID)):
                if num == 1:
                    test_path1.append(path_AR_syn_test + ID + '/' + filename)


        train_path, test_path1, test_path2, test_path3 = np.array(train_path), np.array(test_path1), np.array(test_path2), np.array(test_path3)
        np.random.shuffle(train_path)
        return train_path, test_path1, test_path2, test_path3

    def get_batch_data(self, data, batch_idx, batch_size):
        train_images, ground_truth = [], []
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]

        for path in train_data:
            image = cv2.imread(path, 0) / 255
            if "AR" in path:
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)

            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            train_images.append(image), train_images.append(low1_image), train_images.append(low2_image), train_images.append(low3_image)
            ground_truth.append(image), ground_truth.append(image), ground_truth.append(image), ground_truth.append(image)

        ground_truth = np.array(ground_truth).reshape(-1, 64, 64, 1)
        train_images = np.array(train_images).reshape(-1, 64, 64, 1)
        return ground_truth, train_images

    def train_step(self, low_images, high_images, train=True):
        with tf.GradientTape() as tape:
            z, out1, out2, out3, out4 = self.encoder(low_images)
            _, _, zreg = self.reg(z)
            gen_images = self.generator([zreg, out1, out2, out3, out4])
            image_loss = 10 * tf.reduce_mean(tf.square(high_images - gen_images))
        if train:
            grads = tape.gradient(image_loss, self.generator.trainable_variables + self.encoder.trainable_variables + self.reg.trainable_variables)
            self.g_opti.apply_gradients(zip(grads,  self.generator.trainable_variables + self.encoder.trainable_variables + self.reg.trainable_variables))
        return image_loss


    def training(self):
        image_loss_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            image_loss_batch = []

            for batch in range(self.batch_num):
                high_images, low_images = self.get_batch_data(self.train_path, batch, batch_size=self.batch_size)
                image_loss = self.train_step(low_images, high_images, train=True)
                image_loss_batch.append(image_loss)

            image_loss_epoch.append(np.mean(image_loss_batch))

            print(f'the epoch is {epoch}')
            print(f'the image_loss is {image_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('------------------------------------------------')
            self.generator.save_weights('weights/unet_generator')
            self.test()
            plt.plot(image_loss_epoch, label='Image Loss')
            plt.title('Image Loss')
            plt.savefig('result/GAN/image_loss_unet')
            plt.close()

    def test(self):
        path = '/disk2/bosen/Datasets/Test/'
        psnr, ssim = [[] for i in range(6)], [[] for i in range(6)]
        # for id in os.listdir(path):
        for file_num, filename in enumerate(os.listdir(path)):
                # if file_num == 3:
                #     break
                image = cv2.imread(path + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (11, 11), 0)

                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (20, 20), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low4_image = cv2.resize(cv2.resize(blur_gray, (10, 10), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low5_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                z, out1, out2, out3, out4 = self.encoder(image.reshape(1, 64, 64, 1))
                syn0 = self.generator([z, out1, out2, out3, out4])

                z, out1, out2, out3, out4 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                syn1 = self.generator([z, out1, out2, out3, out4])

                z, out1, out2, out3, out4 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                syn2 = self.generator([z, out1, out2, out3, out4])

                z, out1, out2, out3, out4 = self.encoder(low3_image.reshape(1, 64, 64, 1))
                syn3 = self.generator([z, out1, out2, out3, out4])

                z, out1, out2, out3, out4 = self.encoder(low4_image.reshape(1, 64, 64, 1))
                syn4 = self.generator([z, out1, out2, out3, out4])

                z, out1, out2, out3, out4 = self.encoder(low5_image.reshape(1, 64, 64, 1))
                syn5 = self.generator([z, out1, out2, out3, out4])


                # syn1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                # syn2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                # syn3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
                # syn4 = self.encoder(low4_image.reshape(1, 64, 64, 1))
                # syn5 = self.encoder(low5_image.reshape(1, 64, 64, 1))

                # z0 = self.encoder(image.reshape(1, 64, 64, 1))
                # z1 = self.encoder(low1_image.reshape(1, 64, 64, 1))
                # z2 = self.encoder(low2_image.reshape(1, 64, 64, 1))
                # z3 = self.encoder(low3_image.reshape(1, 64, 64, 1))
                # z4 = self.encoder(low4_image.reshape(1, 64, 64, 1))
                # z5 = self.encoder(low5_image.reshape(1, 64, 64, 1))

                # _, _, zreg0 = self.reg(z0)
                # _, _, zreg1 = self.reg(z1)
                # _, _, zreg2 = self.reg(z2)
                # _, _, zreg3 = self.reg(z3)
                # _, _, zreg4 = self.reg(z4)
                # _, _, zreg5 = self.reg(z5)

                # syn00 = self.generator(zreg0)
                # syn11 = self.generator(zreg1)
                # syn22 = self.generator(zreg2)
                # syn33 = self.generator(zreg3)
                # syn44 = self.generator(zreg4)
                # syn55 = self.generator(zreg5)

                # plt.subplots(figsize=(2, 7))
                # plt.subplots_adjust(wspace=0, hspace=0)
                #
                # plt.subplot(7, 2, 1)
                # plt.axis('off')
                # plt.imshow(image, cmap='gray')
                # plt.subplot(7, 2, 3)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn0, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 5)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn1, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 7)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn2, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 9)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn3, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 11)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn4, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 13)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn5, [64, 64]), cmap='gray')
                #
                # plt.subplot(7, 2, 2)
                # plt.axis('off')
                # plt.imshow(image, cmap='gray')
                # plt.subplot(7, 2, 4)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn00, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 6)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn11, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 8)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn22, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 10)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn33, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 12)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn44, [64, 64]), cmap='gray')
                # plt.subplot(7, 2, 14)
                # plt.axis('off')
                # plt.imshow(tf.reshape(syn55, [64, 64]), cmap='gray')
                # plt.show()

                psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                psnr[4].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                psnr[5].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

                ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn0, dtype=tf.float32), max_val=1)[0])
                ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32), max_val=1)[0])
                ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32), max_val=1)[0])
                ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32), max_val=1)[0])
                ssim[4].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32), max_val=1)[0])
                ssim[5].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn5, dtype=tf.float32), max_val=1)[0])

        psnr, ssim = np.array(psnr), np.array(ssim)
        psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
        print(psnr)
        print(ssim)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # AE = PatchGAN(epochs=50, batch_num=69, batch_size=60)
    # # AE.training()
    # AE.test()

    # unet = Unet(epochs=50, batch_num=69, batch_size=60)
    # unet.training()
    # unet.test()

    encoder = encoder()
    decoder = decoder()

    encoder.load_weights('weights/encoder')
    decoder.load_weights('weights/decoder')

    path = '/disk2/bosen/Datasets/AR_test/'
    psnr, ssim = [[] for i in range(4)], [[] for i in range(4)]
    for id in os.listdir(path):
        for file_num, filename in enumerate(os.listdir(path + id)):
            if file_num == 1:
                break
            image = cv2.imread(path + id + '/' + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

            low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
            low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

            z, _, _, _, _ = encoder(image.reshape(1, 64, 64, 1))
            syn1 = decoder(z)

            z, _, _, _, _ = encoder(low1_image.reshape(1, 64, 64, 1))
            syn2 = decoder(z)

            z, _, _, _, _ = encoder(low2_image.reshape(1, 64, 64, 1))
            syn3 = decoder(z)

            z, _, _, _, _ = encoder(low3_image.reshape(1, 64, 64, 1))
            syn4 = decoder(z)


            psnr[0].append(tf.image.psnr(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32),max_val=1)[0])
            psnr[1].append(tf.image.psnr(tf.cast(tf.reshape(low1_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32),max_val=1)[0])
            psnr[2].append(tf.image.psnr(tf.cast(tf.reshape(low2_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32),max_val=1)[0])
            psnr[3].append(tf.image.psnr(tf.cast(tf.reshape(low3_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32),max_val=1)[0])

            ssim[0].append(tf.image.ssim(tf.cast(tf.reshape(image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn1, dtype=tf.float32),max_val=1)[0])
            ssim[1].append(tf.image.ssim(tf.cast(tf.reshape(low1_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn2, dtype=tf.float32),max_val=1)[0])
            ssim[2].append(tf.image.ssim(tf.cast(tf.reshape(low2_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn3, dtype=tf.float32),max_val=1)[0])
            ssim[3].append(tf.image.ssim(tf.cast(tf.reshape(low3_image, [1, 64, 64, 1]), dtype=tf.float32), tf.cast(syn4, dtype=tf.float32),max_val=1)[0])

    psnr, ssim = np.array(psnr), np.array(ssim)
    psnr, ssim = tf.reduce_mean(psnr, axis=-1), tf.reduce_mean(ssim, axis=-1)
    print(psnr)
    print(ssim)



# tf.Tensor([28.847378 35.706177 37.08207  40.65654 ], shape=(4,), dtype=float32)
# tf.Tensor([0.9299611  0.975621   0.98500234 0.9930656 ], shape=(4,), dtype=float32)








