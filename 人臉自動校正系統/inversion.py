from train_warp3 import *
import timeit

class test_inversion():
    def __init__(self, epochs, warpN, num, reference_num, filename):
        # set the parameters
        self.epochs = epochs
        self.learning_rate = 1e-3
        self.warpN = warpN
        self.num = num
        self.filename = filename
        self.reference_num = reference_num

        # prepare the model
        self.vgg_feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
        self.cls = load_model("/home/bosen/gradation_thesis/recognition_system/model_weights/cls_ca_loss_alignment.h5")
        self.last_conv_layer_name = 'conv2d_3'
        self.network_layer_name_cls = ['max_pooling2d_1', 'conv2d_4', 'conv2d_5', 'max_pooling2d_2', 'conv2d_6', 'conv2d_7', 'max_pooling2d_3', 'flatten', 'dense', 'dropout', 'dense_1']
        self.geometry = load_model('model_weight/geometry-Lce+Limg-warp3.h5')
        self.network_layer_name = ['dense_1']
        self.last_layer_name = 'dense'
        self.last_layer = self.geometry.get_layer(self.last_layer_name)
        network_input = Input(shape=self.last_layer.output.shape[1:])
        x = network_input
        for layer_name in self.network_layer_name:
            x = self.geometry.get_layer(layer_name)(x)
        self.last_layer_model = Model(self.geometry.inputs, self.last_layer.output)
        self.feature_extraction = Model(network_input, x)

        # prepare the data
        self.test_path, self.test_reference, self.test_label = load_test_data_path(shuffle=False)
        _, self.test_reference_image, _, _ = get_batch_data(self.test_path, self.test_reference, self.test_label, self.reference_num, 12)
        self.test_image, _, self.test_points, self.test_label = get_batch_data(self.test_path, self.test_reference, self.test_label, num, 12)

        if self.num == 0:
            self.test_image = [list(self.test_image[3])] + [list(self.test_image[4])] + [list(self.test_image[8])]
            self.test_reference_image = [list(self.test_reference_image[3])] + [list(self.test_reference_image[4])] + [list(self.test_reference_image[8])]
            self.test_points = [list(self.test_points[3])] + [list(self.test_points[4])] + [list(self.test_points[8])]
            self.test_label = [list(self.test_label[3])] + [list(self.test_label[4])] + [list(self.test_label[8])]
        if self.num == 104:
            self.test_image = [list(self.test_image[1])] + [list(self.test_image[4])] + [list(self.test_image[10])]
            self.test_reference_image = [list(self.test_reference_image[1])] + [list(self.test_reference_image[4])] + [list(self.test_reference_image[10])]
            self.test_points = [list(self.test_points[1])] + [list(self.test_points[4])] + [list(self.test_points[10])]
            self.test_label = [list(self.test_label[1])] + [list(self.test_label[4])] + [list(self.test_label[10])]
        if self.num == 117:
            self.test_image = [list(self.test_image[5])] + [list(self.test_image[7])] + [list(self.test_image[10])]
            self.test_reference_image = [list(self.test_reference_image[5])] + [list(self.test_reference_image[7])] + [list(self.test_reference_image[10])]
            self.test_points = [list(self.test_points[5])] + [list(self.test_points[7])] + [list(self.test_points[10])]
            self.test_label = [list(self.test_label[5])] + [list(self.test_label[7])] + [list(self.test_label[10])]
        self.test_image, self.test_reference_image, self.test_points, self.test_label = np.array(self.test_image), np.array(self.test_reference_image), np.array(self.test_points), np.array(self.test_label)
        self.refMtrx = self.find_refMtrx(self.test_points)

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

    def heatmaps_segmentation(self, heatmaps):
        heatmaps = tf.where((0 <= heatmaps) & (heatmaps <= 0.2), x=0, y=heatmaps)
        heatmaps = tf.where((0.2 < heatmaps) & (heatmaps <= 0.4), x=0.25, y=heatmaps)
        heatmaps = tf.where((0.4 < heatmaps) & (heatmaps <= 0.6), x=0.5, y=heatmaps)
        heatmaps = tf.where((0.6 < heatmaps) & (heatmaps <= 0.8), x=0.75, y=heatmaps)
        heatmaps = tf.where((0.8 < heatmaps) & (heatmaps <= 1), x=1, y=heatmaps)
        return heatmaps

    def calculate_loss_acc(self, gt_image,  pred_image, gt_cam, pred_cam, y_true, y_pred):
        gt_image, pred_image = tf.cast(gt_image, dtype="float32"), tf.cast(pred_image, dtype="float32")
        gt_cam, pred_cam = tf.cast(gt_cam, dtype='float32'), tf.cast(pred_cam, dtype='float32')
        gt_image = tf.image.grayscale_to_rgb(gt_image)
        pred_image = tf.image.grayscale_to_rgb(pred_image)
        gt_cam = tf.image.grayscale_to_rgb(gt_cam)
        pred_cam = tf.image.grayscale_to_rgb(pred_cam)

        gt_feature = self.vgg_feature_extraction(gt_image)
        pred_feature = self.vgg_feature_extraction(pred_image)
        gt_cam_feature = self.vgg_feature_extraction(gt_cam)
        pred_cam_feature = self.vgg_feature_extraction(pred_cam)

        ce_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        image_loss = tf.reduce_mean(tf.square(gt_image - pred_image))
        style_loss = tf.reduce_mean(tf.square(gt_feature - pred_feature))
        attention_loss = tf.reduce_mean(tf.square(gt_cam_feature - pred_cam_feature))
        return 40*ce_loss, 15 * attention_loss, 60 * image_loss, 25 * style_loss

    def inversion_step(self, test_image, test_reference, inv_p, label, normalization=None):
        with tf.GradientTape(persistent=True) as p_tape:
            p_tape.watch(inv_p)
            pMtrx = vec2mtrx(len(self.refMtrx), inv_p)
            pMtrx = tf.cast(pMtrx, dtype=tf.float32)
            final_imagewarp = []
            for i in range(len(self.test_points)):
                warp = transformImage(tf.reshape(test_image[i], [1, 192, 256, 3]), tf.cast(self.refMtrx[i], dtype=tf.float32), pMtrx[i], 1, self.test_points[i][2], self.test_points[i][3])
                final_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
            final_imagewarp = tf.cast(final_imagewarp, dtype=tf.float32)
            pred_reference = self.cls(test_reference)
            pred_test = self.cls(final_imagewarp)
            _, heatmaps_pred = gradcam_heatmap_mutiple(final_imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name_cls, label)
            _, heatmaps_reference = gradcam_heatmap_mutiple(test_reference, self.cls, self.last_conv_layer_name, self.network_layer_name_cls, label)
            heatmaps_reference = self.heatmaps_segmentation(heatmaps_reference)
            heatmaps_pred = self.heatmaps_segmentation(heatmaps_pred)
            ce_loss, attention_loss, image_loss, style_loss = self.calculate_loss_acc(test_reference, final_imagewarp, heatmaps_reference, heatmaps_pred, pred_test, pred_reference)

            if normalization is None:
                total_loss = attention_loss

            if normalization is not None:
                # normalized the style_loss
                if image_loss > normalization[0][0]:
                    normalization[0][0] = image_loss
                if image_loss < normalization[0][1]:
                    normalization[0][1] = image_loss
                # normalized the distillation_loss
                if attention_loss > normalization[1][0]:
                    normalization[1][0] = attention_loss
                if attention_loss < normalization[1][1]:
                    normalization[1][1] = attention_loss
                # normalized the ID loss
                if style_loss > normalization[2][0]:
                    normalization[2][0] = style_loss
                if style_loss < normalization[2][1]:
                    normalization[2][1] = style_loss

                image_loss_normalization = ((image_loss - normalization[0][1]) / (normalization[0][0] - normalization[0][1]))
                attention_loss_normalization = ((attention_loss - normalization[1][1]) / (normalization[1][0] - normalization[1][1]))
                style_loss_normalization = ((style_loss - normalization[2][1]) / (normalization[2][0] - normalization[2][1]))
                total_loss = attention_loss_normalization

        if normalization is None:
            gradient_p = p_tape.gradient(total_loss, inv_p)
            inv_p = inv_p - self.learning_rate * gradient_p
            return image_loss, style_loss, attention_loss, inv_p

        if normalization is not None:
            gradient_p = p_tape.gradient(total_loss, inv_p)
            inv_p = inv_p - self.learning_rate * gradient_p
            return image_loss_normalization, style_loss_normalization, attention_loss_normalization, inv_p, normalization

    def p_inversion(self):
        test_image = tf.cast(self.test_image, dtype=tf.float32)
        test_image = tf.image.grayscale_to_rgb(test_image)
        p = tf.tile(tf.constant([[0, 0, 0, 0, 0, 0]], dtype='float32'), [len(self.refMtrx), 1])

        for l in range(self.warpN):
            pMtrx = vec2mtrx(len(self.refMtrx), p)
            imagewarp = []
            for i in range(len(self.test_points)):
                warp = transformImage(tf.reshape(test_image[i], [1, 192, 256, 3]), tf.cast(self.refMtrx[i], dtype=tf.float32), pMtrx[i], 1, self.test_points[i][2], self.test_points[i][3])
                imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
            imagewarp = tf.cast(imagewarp, dtype=tf.float32)
            if l == 0:
                cvwarp = imagewarp
            imagewarp = tf.image.grayscale_to_rgb(imagewarp)
            feat = self.geometry(imagewarp)
            dp = feat
            p = compose(len(self.refMtrx), p, dp)

        pMtrx = vec2mtrx(len(self.refMtrx), p)
        pMtrx = tf.cast(pMtrx, dtype=tf.float32)
        forward_imagewarp = []
        for i in range(len(self.test_points)):
            warp = transformImage(tf.reshape(test_image[i], [1, 192, 256, 3]), tf.cast(self.refMtrx[i], dtype=tf.float32), pMtrx[i], 1, self.test_points[i][2], self.test_points[i][3])
            forward_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
        forward_imagewarp = tf.cast(forward_imagewarp, dtype=tf.float32)

        loss_normalization = [[] for i in range(3)]
        img_loss, style_loss, att_loss = [], [], []
        img_loss_nor, style_loss_nor, att_loss_nor = [], [], []
        for epoch in range(1, self.epochs+1):
            if epoch <= 50:
                start = time.time()
                limg, lstyle, latt, p = self.inversion_step(test_image, self.test_reference_image, p, self.test_label)
                img_loss.append(limg)
                style_loss.append(lstyle)
                att_loss.append(latt)

                print('---------------------------')
                print(f'the epoch is {epoch}')
                print(f'the img loss is {limg}')
                print(f'the style loss is {lstyle}')
                print(f'the att los is {latt}')
                print(f"the new_code is {p[0]}")
                print(f'the spend time is {time.time() - start} second')

            if epoch == 51:
                loss_normalization[0].append(max(img_loss[0: 50])), loss_normalization[0].append(min(img_loss[0: 50]))
                loss_normalization[1].append(max(style_loss[0: 50])), loss_normalization[1].append(min(style_loss[0: 50]))
                loss_normalization[2].append(max(att_loss[0: 50])), loss_normalization[2].append(min(att_loss[0: 50]))

            if epoch > 51:
                limg, lstyle, latt, p, loss_normalization = self.inversion_step(test_image, self.test_reference_image, p, self.test_label, normalization=loss_normalization)
                img_loss_nor.append(limg)
                style_loss_nor.append(lstyle)
                att_loss_nor.append(latt)

                print('---------------------------')
                print(f'the epoch is {epoch}')
                print(f'the img loss is {limg}')
                print(f'the style loss is {lstyle}')
                print(f'the att los is {latt}')
                print(f"the new_code is {p[0]}")
                print(f'the spend time is {time.time() - start} second')
                if limg == 0 or epoch > 50:
                    self.plot_sample_test(epoch, p, cvwarp, forward_imagewarp)

        plt.plot(img_loss_nor)
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.savefig(f'result/{self.filename}/img_loss_{self.num}_diff_re.jpg')
        plt.close()

        plt.plot(style_loss_nor)
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.savefig(f'result/{self.filename}/style_loss_{self.num}_diff_re.jpg')
        plt.close()

        plt.plot(att_loss_nor)
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.savefig(f'result/{self.filename}/att_loss_{self.num}_diff_re.jpg')
        plt.close()

    def plot_sample_test(self, epoch, inv_p, cvwarp, forward_imagewarp):
        test_image = tf.cast(self.test_image, dtype=tf.float32)
        test_image = tf.image.grayscale_to_rgb(test_image)
        pMtrx = vec2mtrx(len(self.test_points), inv_p)
        pMtrx = tf.cast(pMtrx, dtype=tf.float32)
        final_imagewarp = []
        for i in range(len(self.test_points)):
            warp = transformImage(tf.reshape(test_image[i], [1, 192, 256, 3]), tf.cast(self.refMtrx[i], dtype=tf.float32), pMtrx[i], 1, self.test_points[i][2], self.test_points[i][3])
            final_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
        final_imagewarp = tf.cast(final_imagewarp, dtype=tf.float32)

        _, heatmaps_cvwarp = gradcam_heatmap_mutiple(cvwarp, self.cls, self.last_conv_layer_name, self.network_layer_name_cls, self.test_points, corresponding_label=False)
        _, heatmaps_reference = gradcam_heatmap_mutiple(self.test_reference_image, self.cls, self.last_conv_layer_name, self.network_layer_name_cls, self.test_points, corresponding_label=False)
        _, heatmaps_forward_imagewarp = gradcam_heatmap_mutiple(forward_imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name_cls, self.test_points, corresponding_label=False)
        _, heatmaps_final_imagewarp = gradcam_heatmap_mutiple(final_imagewarp, self.cls, self.last_conv_layer_name, self.network_layer_name_cls, self.test_points, corresponding_label=False)

        heatmaps_cvwarp_seg = self.heatmaps_segmentation(heatmaps_cvwarp)
        heatmaps_reference_seg = self.heatmaps_segmentation(heatmaps_reference)
        heatmaps_forward_imagewarp_seg = self.heatmaps_segmentation(heatmaps_forward_imagewarp)
        heatmaps_final_imagewarp_seg = self.heatmaps_segmentation(heatmaps_final_imagewarp)

        plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(len(self.test_image)):
            plt.subplot(8, 12, i + 1)
            plt.axis('off')
            plt.imshow(heatmaps_cvwarp[i], cmap='gray')

            plt.subplot(8, 12, i + 13)
            plt.axis('off')
            plt.imshow(heatmaps_cvwarp_seg[i], cmap='gray')

            plt.subplot(8, 12, i + 25)
            plt.axis('off')
            plt.imshow(heatmaps_forward_imagewarp[i], cmap='gray')

            plt.subplot(8, 12, i + 37)
            plt.axis('off')
            plt.imshow(heatmaps_forward_imagewarp_seg[i], cmap='gray')

            plt.subplot(8, 12, i + 49)
            plt.axis('off')
            plt.imshow(heatmaps_final_imagewarp[i], cmap='gray')

            plt.subplot(8, 12, i + 61)
            plt.axis('off')
            plt.imshow(heatmaps_final_imagewarp_seg[i], cmap='gray')

            plt.subplot(8, 12, i + 73)
            plt.axis('off')
            plt.imshow(heatmaps_reference[i], cmap='gray')

            plt.subplot(8, 12, i + 85)
            plt.axis('off')
            plt.imshow(heatmaps_reference_seg[i], cmap='gray')
        plt.savefig(f'result/{self.filename}/{epoch}-{self.num}-CAM_diff_re.jpg')
        plt.close()


        with open(f'result/{self.filename}/{epoch}-metrics-record-{self.num}_diff_re.csv', 'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(self.test_reference_image)):
                init_warp = cvwarp[i]
                writer.writerow([
                                    f'{tf.image.psnr(tf.cast(tf.reshape(self.test_reference_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(init_warp, [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                    f'{tf.image.ssim(tf.cast(tf.reshape(self.test_reference_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(init_warp, [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                    f'{tf.image.psnr(tf.cast(tf.reshape(self.test_reference_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(forward_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                    f'{tf.image.ssim(tf.cast(tf.reshape(self.test_reference_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(forward_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                    f'{tf.image.psnr(tf.cast(tf.reshape(self.test_reference_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(final_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}',
                                    f'{tf.image.ssim(tf.cast(tf.reshape(self.test_reference_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(final_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}'])

        cvwarp = tf.image.grayscale_to_rgb(cvwarp)
        forward_imagewarp = tf.image.grayscale_to_rgb(forward_imagewarp)
        final_imagewarp = tf.image.grayscale_to_rgb(final_imagewarp)

        plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(len(self.test_image)):
            plt.subplot(5, 12, i + 1)
            plt.axis('off')
            plt.imshow(test_image[i], cmap='gray')
            plt.subplot(5, 12, i + 13)
            plt.axis('off')
            plt.imshow(cvwarp[i], cmap='gray')
            plt.subplot(5, 12, i + 25)
            plt.axis('off')
            plt.imshow(forward_imagewarp[i], cmap='gray')
            plt.subplot(5, 12, i + 37)
            plt.axis('off')
            plt.imshow(final_imagewarp[i], cmap='gray')
            plt.subplot(5, 12, i + 49)
            plt.axis('off')
            plt.imshow(self.test_reference_image[i], cmap='gray')
        plt.savefig(f'result/{self.filename}/{epoch}-{self.num}_diff_re.jpg')
        plt.close()


if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.55
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    test_inversion = test_inversion(epochs=1000, warpN=3, num=0, reference_num=104, filename='p_inversion_diff_reference/warp3_model/Limg+Lstyle+Latt/warp3_times')
    test_inversion.p_inversion()






