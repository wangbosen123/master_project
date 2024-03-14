from train_warp1 import *

class correlation_system2():
    def __init__(self, epochs, batch_num, batch_size, filename):
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.geometry = geometry_predictor()
        self.filename = filename
        self.geometry = load_model(f"/home/bosen/gradation_thesis/correlation_system/model_weight/geometry-Lce+Limg-warp1.h5")
        self.geometry = load_model(f'model_weight/geometry-Lce+Limg-warp3.h5')
        self.cls = load_model("/home/bosen/gradation_thesis/recognition_system/model_weights/cls_ca_loss_alignment.h5")
        self.opti = tf.keras.optimizers.Adam(1e-4)
        self.last_conv_layer_name = 'conv2d_3'
        self.network_layer_name = ['max_pooling2d_1', 'conv2d_4', 'conv2d_5', 'max_pooling2d_2', 'conv2d_6', 'conv2d_7', 'max_pooling2d_3', 'flatten', 'dense', 'dropout', 'dense_1']
        self.feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
        self.train_path, self.gt_path, self.label = load_data_path(shuffle=True)
        print(self.train_path.shape, self.gt_path.shape, self.label.shape)
        self.warpN = 3

    def find_refMtrx(self, points):
        canon4pts = [[] for i in range(len(points))]
        image4pts = [[] for i in range(len(points))]
        refMtrx = [[] for i in range(len(points))]
        for num, point in enumerate(points):
            canon4pts[num].append([0, 0])
            canon4pts[num].append([0, point[2]])
            canon4pts[num].append([point[3], point[2]])
            canon4pts[num].append([point[3], 0])

            image4pts[num].append([point[0], point[1]])
            image4pts[num].append([point[0], point[1] + point[2]])
            image4pts[num].append([point[0] + point[3], point[1] + point[2]])
            image4pts[num].append([point[0] + point[3], point[1]])

        image4pts = np.array(image4pts)
        image4pts = image4pts.reshape(-1, 4, 2)
        canon4pts = np.array(canon4pts)
        canon4pts = canon4pts.reshape(-1, 4, 2)

        for num in range(len(points)):
            refMtrx[num].append(fit(Xsrc=canon4pts[num], Xdst=image4pts[num]))
        refMtrx = np.array(refMtrx)
        return refMtrx

    def calculate_loss_acc(self, gt_image,  pred_image, gt_cam, pred_cam, y_true, y_pred):
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        ce_loss = tf.reduce_mean(ce_loss)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        acc = accuracy_score(y_true, y_pred)
        image_loss = tf.reduce_mean(tf.square(gt_image - pred_image))

        gt_image, pred_image = tf.cast(gt_image, dtype="float32"), tf.cast(pred_image, dtype="float32")
        gt_cam, pred_cam = tf.cast(gt_cam, dtype='float32'), tf.cast(pred_cam, dtype='float32')
        gt_image = tf.image.grayscale_to_rgb(gt_image)
        pred_image = tf.image.grayscale_to_rgb(pred_image)
        gt_cam = tf.image.grayscale_to_rgb(gt_cam)
        pred_cam = tf.image.grayscale_to_rgb(pred_cam)

        gt_feature = self.feature_extraction(gt_image)
        pred_feature = self.feature_extraction(pred_image)
        gt_cam_feature = self.feature_extraction(gt_cam)
        pred_cam_feature = self.feature_extraction(pred_cam)
        style_loss = tf.reduce_mean(tf.square(gt_feature - pred_feature))
        attention_loss = tf.reduce_mean(tf.square(gt_cam_feature - pred_cam_feature))
        return ce_loss, 15 * attention_loss, 20 * image_loss, 10 * style_loss, acc

    def train_step(self, train_image, gt_image, points, label):
        total_ce_loss, total_image_loss, total_style_loss, total_attention_loss = 0, 0, 0, 0
        with tf.GradientTape() as tape:
            train_image = tf.cast(train_image, dtype=tf.float32)
            train_image = tf.image.grayscale_to_rgb(train_image)
            refMtrx = self.find_refMtrx(points)
            p = tf.tile(tf.constant([[0, 0, 0, 0, 0, 0]], dtype='float32'), [len(refMtrx), 1])

            for l in range(self.warpN):
                pMtrx = vec2mtrx(len(refMtrx), p)
                imagewarp = []
                for i in range(len(points)):
                    warp = transformImage(tf.reshape(train_image[i], [1, 192, 256, 3]), tf.cast(refMtrx[i], dtype=tf.float32), pMtrx[i], 1, points[i][2], points[i][3])
                    imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
                imagewarp = tf.cast(imagewarp, dtype=tf.float32)

                if l > 0:
                    pred = self.cls(imagewarp)
                    _, heatmaps_pred = gradcam_heatmap_mutiple(imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name, label)
                    _, heatmaps_gt = gradcam_heatmap_mutiple(gt_image, self.cls, self.last_conv_layer_name, self.network_layer_name, label)
                    ce_loss, attention_loss, image_loss, style_loss, acc = self.calculate_loss_acc(gt_image, imagewarp, heatmaps_gt, heatmaps_pred, label, pred)
                    total_ce_loss += ce_loss
                    total_image_loss += image_loss
                    total_style_loss += style_loss
                    total_attention_loss += attention_loss

                imagewarp = tf.image.grayscale_to_rgb(imagewarp)
                feat = self.geometry(imagewarp)
                dp = feat
                p = compose(len(refMtrx), p, dp)

            pMtrx = vec2mtrx(len(refMtrx), p)
            pMtrx = tf.cast(pMtrx, dtype=tf.float32)
            final_imagewarp = []
            for i in range(len(points)):
                warp = transformImage(tf.reshape(train_image[i], [1, 192, 256, 3]), tf.cast(refMtrx[i], dtype=tf.float32), pMtrx[i], 1, points[i][2], points[i][3])
                final_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
            final_imagewarp = tf.cast(final_imagewarp, dtype=tf.float32)

            pred = self.cls(final_imagewarp)
            _, heatmaps_pred = gradcam_heatmap_mutiple(final_imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name, label)
            _, heatmaps_gt = gradcam_heatmap_mutiple(gt_image, self.cls, self.last_conv_layer_name, self.network_layer_name, label)
            ce_loss, attention_loss, image_loss, style_loss, acc = self.calculate_loss_acc(gt_image, final_imagewarp, heatmaps_gt, heatmaps_pred, label, pred)
            total_ce_loss += ce_loss
            total_image_loss += image_loss
            total_style_loss += style_loss
            total_attention_loss += attention_loss
            total_loss = total_ce_loss + total_attention_loss + total_image_loss + total_style_loss
        gradients = tape.gradient(total_loss, self.geometry.trainable_variables)
        self.opti.apply_gradients(zip(gradients, self.geometry.trainable_variables))
        return ce_loss, attention_loss, image_loss, style_loss, acc

    def training(self, start_epoch):
        ce_loss_epoch = []
        attention_loss_epoch = []
        image_loss_epoch = []
        style_loss_epoch = []
        acc_epoch = []

        for epoch in range(start_epoch, self.epochs+1):
            start = time.time()
            ce_loss_batch = []
            attention_loss_batch = []
            image_loss_batch = []
            style_loss_batch = []
            acc_batch = []
            for batch in range(self.batch_num):
                print(batch, end='')
                batch_train_image, batch_gt_image, batch_points, batch_label = get_batch_data(self.train_path, self.gt_path, self.label, batch, self.batch_size)
                ce_loss, attention_loss, image_loss, style_loss, acc = self.train_step(batch_train_image, batch_gt_image, batch_points, batch_label)
                ce_loss_batch.append(ce_loss)
                attention_loss_batch.append(attention_loss)
                image_loss_batch.append(image_loss)
                style_loss_batch.append(style_loss)
                acc_batch.append(acc)

            ce_loss_epoch.append(np.mean(ce_loss_batch))
            attention_loss_epoch.append(np.mean(attention_loss_batch))
            image_loss_epoch.append(np.mean(image_loss_batch))
            style_loss_epoch.append(np.mean(style_loss_batch))
            acc_epoch.append(np.mean(acc_batch))

            print(' ')
            print('___________________________________________________')
            print(f'the epoch is {epoch}')
            print(f'the ce loss is {ce_loss_epoch[-1]}')
            print(f'the attention loss is {attention_loss_epoch[-1]}')
            print(f'the image loss is {image_loss_epoch[-1]}')
            print(f'the style loss is {style_loss_epoch[-1]}')
            print(f'the accuracy is {acc_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print(ce_loss_epoch, attention_loss_epoch, image_loss_epoch, style_loss_epoch, acc_epoch)
            self.geometry.save(f'model_weight/geometry-{self.filename}.h5')

            for i in [0, 13, 26, 39, 52, 65, 78, 91, 104, 117]:
                for j in range(1, 6):
                    self.plot_sample_train(epoch, num=i, warp_times=j, shuffle_default=True)
                    self.plot_sample_train(epoch, num=i, warp_times=j, shuffle_default=False)
                    self.plot_sample_test(epoch, num=i, warp_times=j, shuffle_default=True)
                    self.plot_sample_test(epoch, num=i, warp_times=j, shuffle_default=False)


            if epoch == 40:
                plt.plot(ce_loss_epoch)
                plt.grid(True)
                plt.savefig(f'result/{self.filename}/ce_loss.jpg')
                plt.close()

                plt.plot(attention_loss_epoch)
                plt.grid(True)
                plt.savefig(f'result/{self.filename}/attention_loss.jpg')
                plt.close()

                plt.plot(style_loss_epoch)
                plt.grid(True)
                plt.savefig(f'result/{self.filename}/style_loss.jpg')
                plt.close()

                plt.plot(acc_epoch)
                plt.grid(True)
                plt.savefig(f'result/{self.filename}/acc.jpg')
                plt.close()

                plt.plot(image_loss_epoch)
                plt.grid(True)
                plt.savefig(f'result/{self.filename}/image_loss.jpg')
                plt.close()

    def plot_sample_train(self, epoch, num, warp_times, shuffle_default=False):
        train_path, gt_path, label_path = load_data_path(shuffle=shuffle_default)
        batch_train_image, batch_gt_image, batch_points, batch_label = get_batch_data(train_path, gt_path, label_path, num, 12)
        p = tf.tile(tf.constant([[0, 0, 0, 0, 0, 0]], dtype='float32'), [len(batch_points), 1])
        batch_train_image = tf.cast(batch_train_image, dtype=tf.float32)
        batch_train_image = tf.image.grayscale_to_rgb(batch_train_image)
        batch_refMtrx = self.find_refMtrx(batch_points)

        for l in range(warp_times):
            pMtrx = vec2mtrx(len(batch_points), p)
            imagewarp = []
            for i in range(len(batch_points)):
                warp = transformImage(tf.reshape(batch_train_image[i], [1, 192, 256, 3]), tf.cast(batch_refMtrx[i], dtype=tf.float32), pMtrx[i], 1, batch_points[i][2], batch_points[i][3])
                imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
            imagewarp = tf.cast(imagewarp, dtype=tf.float32)
            if l == 0:
                _, heatmaps_train = gradcam_heatmap_mutiple(imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name, batch_label)
            imagewarp = tf.image.grayscale_to_rgb(imagewarp)
            if l == 0:
                cv_warp = imagewarp
            feat = self.geometry(imagewarp)
            dp = feat
            p = compose(len(batch_refMtrx), p, dp)

        pMtrx = vec2mtrx(len(batch_refMtrx), p)
        pMtrx = tf.cast(pMtrx, dtype=tf.float32)
        final_imagewarp = []
        for i in range(len(batch_points)):
            warp = transformImage(tf.reshape(batch_train_image[i], [1, 192, 256, 3]), tf.cast(batch_refMtrx[i], dtype=tf.float32), pMtrx[i], 1, batch_points[i][2], batch_points[i][3])
            final_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
        final_imagewarp = tf.cast(final_imagewarp, dtype=tf.float32)
        _, heatmaps_gt = gradcam_heatmap_mutiple(batch_gt_image, self.cls, self.last_conv_layer_name, self.network_layer_name, batch_label)
        _, heatmaps_imagewarp = gradcam_heatmap_mutiple(final_imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name, batch_label)


        plt.subplots(figsize=(10, 4))
        for i in range(len(batch_train_image)):
            plt.subplot(6, 12, i + 1)
            plt.axis('off')
            plt.imshow(cv_warp[i], cmap='gray')

            plt.subplot(6, 12, i + 13)
            plt.axis('off')
            plt.imshow(heatmaps_train[i], cmap='gray')

            plt.subplot(6, 12, i + 25)
            plt.axis('off')
            plt.imshow(final_imagewarp[i], cmap='gray')

            plt.subplot(6, 12, i + 37)
            plt.axis('off')
            plt.imshow(heatmaps_imagewarp[i], cmap='gray')

            plt.subplot(6, 12, i + 49)
            plt.axis('off')
            plt.imshow(batch_gt_image[i], cmap='gray')

            plt.subplot(6, 12, i + 61)
            plt.axis('off')
            plt.imshow(heatmaps_gt[i], cmap='gray')

        plt.savefig(f'result/{self.filename}/{epoch}_shuffle_{shuffle_default}_{num}_warp_{warp_times}_train.jpg')
        plt.close()
    def plot_sample_test(self, epoch, num, warp_times, shuffle_default=False):
        test_path, gt_path, label_path = load_test_data_path(shuffle=shuffle_default)
        batch_test_image, batch_gt_image, batch_points, batch_label = get_batch_data(test_path, gt_path, label_path, num, 12)
        p = tf.tile(tf.constant([[0, 0, 0, 0, 0, 0]], dtype='float32'), [len(batch_points), 1])
        batch_test_image = tf.cast(batch_test_image, dtype=tf.float32)
        batch_test_image = tf.image.grayscale_to_rgb(batch_test_image)
        batch_refMtrx = self.find_refMtrx(batch_points)


        for l in range(warp_times):
            pMtrx = vec2mtrx(len(batch_points), p)
            imagewarp = []
            for i in range(len(batch_points)):
                warp = transformImage(tf.reshape(batch_test_image[i], [1, 192, 256, 3]), tf.cast(batch_refMtrx[i], dtype=tf.float32), pMtrx[i], 1, batch_points[i][2], batch_points[i][3])
                imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
            imagewarp = tf.cast(imagewarp, dtype=tf.float32)
            if l == 0:
                _, heatmaps_test = gradcam_heatmap_mutiple(imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name, batch_label, corresponding_label=False)
            imagewarp = tf.image.grayscale_to_rgb(imagewarp)
            if l == 0:
                cv_warp = imagewarp
            feat = self.geometry(imagewarp)
            dp = feat
            p = compose(len(batch_refMtrx), p, dp)

        pMtrx = vec2mtrx(len(batch_refMtrx), p)
        pMtrx = tf.cast(pMtrx, dtype=tf.float32)
        final_imagewarp = []
        for i in range(len(batch_points)):
            warp = transformImage(tf.reshape(batch_test_image[i], [1, 192, 256, 3]), tf.cast(batch_refMtrx[i], dtype=tf.float32), pMtrx[i], 1, batch_points[i][2], batch_points[i][3])
            final_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
        final_imagewarp = tf.cast(final_imagewarp, dtype=tf.float32)
        _, heatmaps_gt = gradcam_heatmap_mutiple(batch_gt_image, self.cls, self.last_conv_layer_name, self.network_layer_name, batch_label, corresponding_label=False)
        _, heatmaps_imagewarp = gradcam_heatmap_mutiple(final_imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name, batch_label, corresponding_label=False)


        plt.subplots(figsize=(10, 4))
        for i in range(len(batch_test_image)):
            plt.subplot(6, 12, i + 1)
            plt.axis('off')
            plt.imshow(cv_warp[i], cmap='gray')

            plt.subplot(6, 12, i + 13)
            plt.axis('off')
            plt.imshow(heatmaps_test[i], cmap='gray')

            plt.subplot(6, 12, i + 25)
            plt.axis('off')
            plt.imshow(final_imagewarp[i], cmap='gray')

            plt.subplot(6, 12, i + 37)
            plt.axis('off')
            plt.imshow(heatmaps_imagewarp[i], cmap='gray')

            plt.subplot(6, 12, i + 49)
            plt.axis('off')
            plt.imshow(batch_gt_image[i], cmap='gray')

            plt.subplot(6, 12, i + 61)
            plt.axis('off')
            plt.imshow(heatmaps_gt[i], cmap='gray')

        plt.savefig(f'result/{self.filename}/{epoch}_shuffle_{shuffle_default}_{num}_warp_{warp_times}test.jpg')
        plt.close()
        if shuffle_default == False:
            with open(f'result/{self.filename}/Epoch-{epoch}-{num}-warp-{warp_times}-metrics-record.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(batch_gt_image)):
                    init_warp = tf.image.rgb_to_grayscale(cv_warp[i])
                    writer.writerow([f'{tf.image.psnr(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(init_warp, [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                     f'{tf.image.ssim(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(init_warp, [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                     f'{tf.image.psnr(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(final_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                     f'{tf.image.ssim(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(final_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}'])

if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.55
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    correlation_system = correlation_system2(85, 81, 150, filename='Lce+Limg-warp3')
    correlation_system.training(76)
    # for i in [0, 13, 26, 39, 52, 65, 78, 91, 104, 117]:
    #     correlation_system.plot_sample_train(20, num=i, shuffle_default=True)
    #     correlation_system.plot_sample_train(20, num=i, shuffle_default=True)
    #     correlation_system.plot_sample_test(20, num=i, shuffle_default=True)
    #     correlation_system.plot_sample_test(20, num=i, shuffle_default=True)

    # [8.25018, 6.2415085, 6.09402, 6.159714, 6.128489, 6.1317053, 6.141803, 6.16387, 6.178957, 6.2085686, 6.2675643,6.2532363, 6.27421, 6.2612686, 6.288324, 6.327433, 6.372233, 6.3903136, 6.441678, 6.46495]
    # [1.658172, 1.5813764, 1.5793493, 1.5716448, 1.5656425, 1.5616864, 1.5647501, 1.5630089, 1.56371, 1.5624394, 1.5644282, 1.5588013, 1.5564069, 1.5547316, 1.5533172, 1.5509645, 1.5547217, 1.5526565, 1.5492309, 1.5481514]
    # [2.3332174, 2.2348044, 2.2026565, 2.1909811, 2.1792657, 2.1739578, 2.168809, 2.1657212, 2.1626322, 2.1603658, 2.1570897, 2.1572542, 2.1547046, 2.1530476, 2.1539161, 2.1506524, 2.1508505, 2.1496108, 2.1491866, 2.1496089]
    # [1.4113858, 1.3923762, 1.3935612, 1.3907704, 1.3790432, 1.3714145, 1.3669989, 1.3629683, 1.3593528, 1.3565241, 1.3558322, 1.3535342, 1.3534441, 1.3517739, 1.3536295, 1.3519603, 1.3542906, 1.3558815, 1.3542671, 1.3542958]
    # [0.23125679105838054, 0.2942350949160626, 0.3084401580382212, 0.3192473282039376, 0.3261950679718511, 0.3316016491804317, 0.33475587706063786, 0.3411743777038764, 0.34135026181558636, 0.343420085333657, 0.3435924738864598, 0.3473211096167125, 0.34601566292654873, 0.34773582463489594, 0.34749809058240766, 0.34408911736651704, 0.34392845953736983, 0.34567757146172534, 0.34617738896990063, 0.34668536799489874]


