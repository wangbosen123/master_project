import matplotlib.pyplot as plt
from train_warp1 import *

def find_refMtrx(points):
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

def plot_sample_test(num, warp_model, warp_times, filename, shuffle_default=False):
    last_conv_layer_name = 'conv2d_3'
    network_layer_name = ['max_pooling2d_1', 'conv2d_4', 'conv2d_5', 'max_pooling2d_2', 'conv2d_6', 'conv2d_7', 'max_pooling2d_3', 'flatten', 'dense', 'dropout', 'dense_1']
    cls = load_model("/home/bosen/gradation_thesis/recognition_system/model_weights/cls_ca_loss_alignment.h5")

    test_path, gt_path, label_path = load_test_data_path(shuffle=shuffle_default)
    batch_test_image, batch_gt_image, batch_points, batch_label = get_batch_data(test_path, gt_path, label_path, num, 12)
    p = tf.tile(tf.constant([[0, 0, 0, 0, 0, 0]], dtype='float32'), [len(batch_points), 1])
    batch_test_image = tf.cast(batch_test_image, dtype=tf.float32)
    batch_test_image = tf.image.grayscale_to_rgb(batch_test_image)
    batch_refMtrx = find_refMtrx(batch_points)

    for l in range(warp_times):
        pMtrx = vec2mtrx(len(batch_points), p)
        imagewarp = []
        for i in range(len(batch_points)):
            warp = transformImage(tf.reshape(batch_test_image[i], [1, 192, 256, 3]), tf.cast(batch_refMtrx[i], dtype=tf.float32), pMtrx[i], 1, batch_points[i][2], batch_points[i][3])
            imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
        imagewarp = tf.cast(imagewarp, dtype=tf.float32)
        if l == 0:
            _, heatmaps_test = gradcam_heatmap_mutiple(imagewarp, cls, last_conv_layer_name, network_layer_name, batch_label, corresponding_label=False)
        imagewarp = tf.image.grayscale_to_rgb(imagewarp)
        if l == 0:
            cv_warp = imagewarp
        if l == 1:
            warp1 = imagewarp
        if l == 2:
            warp2 = imagewarp
        if l == 3:
            warp3 = imagewarp
        feat = warp_model(imagewarp)
        dp = feat
        p = compose(len(batch_refMtrx), p, dp)

    pMtrx = vec2mtrx(len(batch_refMtrx), p)
    pMtrx = tf.cast(pMtrx, dtype=tf.float32)
    final_imagewarp = []
    for i in range(len(batch_points)):
        warp = transformImage(tf.reshape(batch_test_image[i], [1, 192, 256, 3]), tf.cast(batch_refMtrx[i], dtype=tf.float32), pMtrx[i], 1, batch_points[i][2], batch_points[i][3])
        final_imagewarp.append(tf.reshape(tf.image.resize(warp, [64, 64], method='bicubic'), [64, 64, 1]))
    final_imagewarp = tf.cast(final_imagewarp, dtype=tf.float32)
    _, heatmaps_gt = gradcam_heatmap_mutiple(batch_gt_image, cls, last_conv_layer_name, network_layer_name, batch_label, corresponding_label=False)
    _, heatmaps_imagewarp = gradcam_heatmap_mutiple(final_imagewarp, cls, last_conv_layer_name, network_layer_name, batch_label, corresponding_label=False)
    final_imagewarp = tf.image.grayscale_to_rgb(final_imagewarp)

    plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(batch_test_image)):
        plt.subplot(7, 12, i + 1)
        plt.axis('off')
        plt.imshow(batch_test_image[i], cmap='gray')

        plt.subplot(7, 12, i + 13)
        plt.axis('off')
        plt.imshow(cv_warp[i], cmap='gray')

        plt.subplot(7, 12, i + 25)
        plt.axis('off')
        plt.imshow(warp1[i], cmap='gray')

        plt.subplot(7, 12, i + 37)
        plt.axis('off')
        plt.imshow(warp2[i], cmap='gray')

        plt.subplot(7, 12, i + 49)
        plt.axis('off')
        plt.imshow(warp3[i], cmap='gray')

        plt.subplot(7, 12, i + 61)
        plt.axis('off')
        plt.imshow(final_imagewarp[i], cmap='gray')

        plt.subplot(7, 12, i + 73)
        plt.axis('off')
        plt.imshow(batch_gt_image[i], cmap='gray')

    plt.savefig(f'result/{filename}shuffle_{shuffle_default}_{num}_warp_{warp_times}test.jpg')
    plt.close()

    # if shuffle_default == False:
    #     with open(f'result/{filename}{num}-warp-{warp_times}-metrics-record.csv', 'w',
    #               newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         for i in range(len(batch_gt_image)):
    #             init_warp = tf.image.rgb_to_grayscale(cv_warp[i])
    #             writer.writerow([f'{tf.image.psnr(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(init_warp, [64, 64, 1]), dtype=tf.float32), max_val=1)}',f'{tf.image.ssim(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(init_warp, [64, 64, 1]), dtype=tf.float32), max_val=1)}',
    #                             f'{tf.image.psnr(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(final_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}',
    #                             f'{tf.image.ssim(tf.cast(tf.reshape(batch_gt_image[i], [64, 64, 1]), dtype=tf.float32), tf.cast(tf.reshape(final_imagewarp[i], [64, 64, 1]), dtype=tf.float32), max_val=1)}'])



if __name__=='__main__':
    pass
    path = "/home/bosen/gradation_thesis/correlation_system_0402/result/p_inversion_diff_reference/warp3_model/Limg+Lstyle/warp3_times/"
    for filename in os.listdir(path):
        # if '104' in filename:
         os.remove(path + filename)

    # warp1_model = load_model('model_weight/geometry-Lce+Limg-warp1.h5')
    # warp3_model = load_model('model_weight/geometry-Lce+Limg-warp3.h5')
    # for i in [0, 104, 117]:
    #     for j in [4]:
    #         plot_sample_test(i, warp1_model, j, 'data-chart/warp1-model-nocam', shuffle_default=False)
    #         plot_sample_test(i, warp3_model, j, 'data-chart/warp3-model-nocam', shuffle_default=False)

    ID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cv_warp_psnr = [9.63, 9.77, 9.40, 13.13, 14.37, 13.56, 9.10, 7.78, 9.69]
    # forward_psnr = [15.08, 13.12, 13.17, 11.14, 16.32, 15.38, 14.06, 9.90, 14.45]
    # inv_psnr = [14.98, 13.90, 16.20, 12.02, 16.84, 15.77, 14.35, 10.03, 15.13]
    #
    cv_warp_ssim = [0.19, 0.24, 0.17, 0.39, 0.30, 0.29, 0.31, 0.28, 0.27]
    # forward_ssim = [0.34, 0.28, 0.31, 0.4, 0.43, 0.42, 0.34, 0.35, 0.40]
    # inv_ssim = [0.30, 0.29, 0.53, 0.46, 0.42, 0.45, 0.38, 0.31, 0.42]

    forward_psnr = [16.24, 15.98, 16.19, 12.81, 17.12, 17.03, 13.35, 8.91, 12.45]
    # inv_psnr = [16.28, 16.55, 16.29, 13.09, 17.72, 17.13, 14.56, 9.76, 14.55]
    #
    forward_ssim = [0.42, 0.55, 0.51, 0.59, 0.48, 0.59, 0.47, 0.36, 0.39]
    # inv_ssim = [0.42, 0.58, 0.51, 0.56, 0.47, 0.59, 0.57, 0.42, 0.38]

    inv_psnr = [16.28, 16.52, 16.33, 13.03, 17.74, 17.14, 14.16, 9.67, 14.85]
    inv_ssim = [0.42, 0.56, 0.53, 0.58, 0.45, 0.59, 0.48, 0.40, 0.48]


    lines = plt.plot(ID, cv_warp_psnr, ID, forward_psnr, ID, inv_psnr)
    plt.setp(lines, marker="o")
    plt.grid(True)
    plt.title('warp1-model')
    plt.xlabel('ID')
    plt.ylabel('PSNR')
    plt.legend(['cv_warp', 'forward', 'inversion'], loc='lower right')
    plt.savefig('result/data-chart/inver-PSNR-model3warp3-deffreference-att')
    plt.close()

    lines = plt.plot(ID, cv_warp_ssim, ID, forward_ssim, ID, inv_ssim)
    plt.setp(lines, marker="o")
    plt.grid(True)
    plt.title('warp1-model')
    plt.xlabel('ID')
    plt.ylabel('SSIM')
    plt.legend(['cv_warp', 'forward', 'inversion'], loc='lower right')
    plt.savefig('result/data-chart/inver-SSIM-model3warp3-deffreference-att')
    plt.close()

    # 9.638739585876465, 0.1926768273115158, 15.082649230957031, 0.34096387028694153, 14.98282241821289, 0.3008686304092407
    # 9.77340030670166, 0.24534212052822113, 13.129289627075195, 0.2843386232852936, 13.906579971313477, 0.2937668263912201
    # 9.401352882385254, 0.1714475303888321, 13.174180030822754, 0.31059324741363525, 16.2098388671875, 0.5311740636825562

    # 13.134716033935547, 0.39050376415252686, 11.147239685058594, 0.41088899970054626, 12.024409294128418, 0.4655248522758484
    # 14.37658405303955, 0.3048034906387329, 16.321868896484375, 0.43097782135009766, 16.84446907043457, 0.42786937952041626
    # 13.5689115524292, 0.29475870728492737, 15.38613224029541, 0.42659640312194824, 15.79700756072998, 0.4501519799232483

    # 9.107200622558594, 0.3100137710571289, 14.065869331359863, 0.34133437275886536, 14.3504638671875, 0.3861338794231415
    # 7.782174587249756, 0.28338679671287537, 9.905601501464844, 0.351043164730072, 10.035113334655762, 0.3124658465385437
    # 9.698952674865723, 0.27849599719047546, 14.455920219421387, 0.4004081189632416, 15.139575004577637, 0.4212767481803894

    # cvwar_psnr = [10.25, 10.57, 10.87, 8.86, 11.64, 14.88, 12.38, 12.97, 12.73]
    # cvwarp_ssim = [0.17, 0.19, 0.20, 0.25, 0.29, 0.34, 0.30, 0.40, 0.41]
    #
    # warp1_1_psnr = [19.71, 16.77, 16.85, 9.89, 13.79, 16.82, 16.80, 18.82, 18.86]
    # warp1_2_psnr = [15.77, 13.42, 15.87, 10.13, 13.24, 14.15, 15.19, 15.33, 15.87]
    # inv_warp1_2_psnr = [20.29, 18.75, 20.02, 10.12, 13.42, 16.68, 15.65, 16.20, 18.37]
    # warp1_3_psnr = [10.86, 12.32, 14.78, 9.73, 12.19, 12.78, 14.19, 14.76, 15.47]
    # warp1_4_psnr = [12.18, 11.29, 12.61, 9.31, 12.18, 12.49, 13.87, 13.64, 15.05]
    #
    # warp1_1_ssim = [0.65, 0.50, 0.51, 0.53, 0.59, 0.59, 0.47, 0.58, 0.64]
    # warp1_2_ssim = [0.34, 0.29, 0.43, 0.41, 0.42, 0.44, 0.35, 0.35, 0.47]
    # inv_warp1_2_ssim = [0.67, 0.64, 0.62, 0.39, 0.46, 0.46, 0.35, 0.37, 0.57]
    # warp1_3_ssim = [0.23, 0.24, 0.40, 0.22, 0.26, 0.37, 0.26, 0.24, 0.42]
    # warp1_4_ssim = [0.27, 0.11, 0.31, 0.19, 0.21, 0.28, 0.25, 0.15, 0.39]
    #
    # warp3_1_psnr = [17.10, 14.63, 16.21, 9.61, 13.35, 16.69, 18.32, 19.45, 17.95]
    # warp3_2_psnr = [19.60, 15.81, 17.56, 9.86, 13.80, 17.23, 18.42, 19.31, 18.49]
    # warp3_3_psnr = [21.41, 19.05, 18.49, 10.03, 13.98, 16.97, 17.42, 18.37, 18.99]
    # inv_warp3_3_psnr = [22.86, 21.18, 18.85, 10.10, 14.00, 18.38, 17.76, 18.77, 20.06]
    # warp3_4_psnr = [20.50, 20.05, 18.11, 10.08, 13.86, 16.22, 16.40, 17.18, 19.09]
    # warp3_5_psnr = [17.61, 19.31, 18.27, 10.07, 13.59, 15.40, 15.45, 16.38, 18.43]
    #
    # warp3_1_ssim = [0.52, 0.38, 0.49, 0.47, 0.56, 0.56, 0.63, 0.71, 0.63]
    # warp3_2_ssim = [0.66, 0.49, 0.54, 0.52, 0.61, 0.60, 0.59, 0.62, 0.64]
    # warp3_3_ssim = [0.80, 0.59, 0.57, 0.52, 0.63, 0.58, 0.52, 0.54, 0.64]
    # inv_warp3_3_ssim = [0.85, 0.74, 0.58, 0.57, 0.64, 0.58, 0.58, 0.57, 0.68]
    # warp3_4_ssim = [0.77, 0.71, 0.57, 0.48, 0.57, 0.54, 0.43, 0.47, 0.62]
    # warp3_5_ssim = [0.56, 0.64, 0.57, 0.42, 0.49, 0.48, 0.37, 0.44, 0.58]

    # lines = plt.plot(ID, cvwar_psnr, ID, warp3_3_psnr, ID, inv_warp3_3_psnr)
    # plt.setp(lines, marker="o")
    # plt.grid(True)
    # plt.title('warp3-model')
    # plt.xlabel('ID')
    # plt.ylabel('PSNR')
    # plt.legend(['cv_warp', 'warp3_3_psnr', 'inv_warp3_3'], loc='lower right')
    # plt.savefig('result/data-chart/inversion-PSNR-modelwarp3')
    # plt.close()
    #
    # lines = plt.plot(ID, cvwar_psnr, ID, warp1_1_psnr, ID, warp1_2_psnr, ID, warp1_3_psnr, ID, warp1_4_psnr)
    # plt.setp(lines, marker="o")
    # plt.grid(True)
    # plt.title('warp1-model')
    # plt.xlabel('ID')
    # plt.ylabel('PSNR')
    # plt.legend(['cv_warp', 'warp1-1times', 'warp1-2times', 'warp1-3times', 'warp1-4times'], loc='lower right')
    # plt.savefig('result/data-chart/warp-times-PSNR-modelwarp1')
    # plt.close()
    #
    # lines = plt.plot(ID, cvwarp_ssim, ID, warp1_1_ssim, ID, warp1_2_ssim, ID, warp1_3_ssim, ID, warp1_4_ssim)
    # plt.setp(lines, marker="o")
    # plt.grid(True)
    # plt.title('warp1-model')
    # plt.xlabel('ID')
    # plt.ylabel('SSIM')
    # plt.legend(['cv_warp', 'warp1-1times', 'warp1-2times', 'warp1-3times', 'warp1-4times'], loc='lower right')
    # plt.savefig('result/data-chart/warp-times-SSIM-modelwarp1')
    # plt.close()
    #
    # lines = plt.plot(ID, cvwar_psnr, ID, warp1_1_psnr, ID, warp3_1_psnr, ID, warp3_2_psnr, ID, warp3_3_psnr, ID, warp3_4_psnr, ID, warp3_5_psnr)
    # plt.setp(lines, marker="o")
    # plt.grid(True)
    # plt.title('warp3-model')
    # plt.xlabel('ID')
    # plt.ylabel('PSNR')
    # plt.legend(['cv_warp', 'warp1-1times', 'warp3-1times', 'warp3-2times', 'warp3-3times', 'warp3-4times', 'warp3-5times'], loc='lower right')
    # plt.savefig('result/data-chart/warp-times-PSNR-modelwarp3')
    # plt.close()

    # lines = plt.plot(ID, cvwarp_ssim, ID, warp1_1_ssim, ID, warp3_1_ssim, ID, warp3_2_ssim, ID, warp3_3_ssim, ID, warp3_4_ssim, ID, warp3_5_ssim)
    # plt.setp(lines, marker="o")
    # plt.grid(True)
    # plt.title('warp3-model')
    # plt.xlabel('ID')
    # plt.ylabel('SSIM')
    # plt.legend(['cv_warp', 'warp1-1times', 'warp3-1times', 'warp3-2times', 'warp3-3times', 'warp3-4times', 'warp3-5times'], loc='lower right')
    # plt.savefig('result/data-chart/warp-times-SSIM-modelwarp3')
    # plt.close()



# 0(4,5,9)
# 104(2,5,11)
# 107(6,11,8)
# ID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# warp0_psnr = [10.25, 10.57, 10.87, 8.86, 11.64, 14.88, 12.38, 12.97, 12.73]
# warp1_psnr = [19.27, 17.19, 17.03, 9.93, 13.90, 16.61, 17.39, 18.52, 19.43]
# warp2_psnr = [17.35, 17.75, 17.13, 10.14, 14.00, 14.91, 16.65, 16.40, 18.62]
# warp3_psnr = [16.41, 17.71, 06.07,10.12, 13.94, 14.12, 17.85, 17.82, 18.04]
#
# warp0_ssim = [0.17, 0.19, 0.20, 0.25, 0.29, 0.34, 0.30, 0.40, 0.41]
# warp1_ssim = [0.68, 0.58, 0.53, 0.55, 0.69, 0.60, 0.51, 0.59, 0.68]
# warp2_ssim = [0.52, 0.51, 0.54, 0.46, 0.67, 0.50, 0.44, 0.47, 0.64]
# warp3_ssim = [0.62, 0.68, 0.50, 0.45, 0.64, 0.44, 0.52, 0.52, 0.62]
#
# lines = plt.plot(ID, warp0_psnr, ID, warp1_psnr, ID, warp2_psnr, ID, warp3_psnr)
# plt.setp(lines, marker="o")
# plt.grid(True)
# plt.xlabel('ID')
# plt.ylabel('PSNR')
# plt.legend(['cv_warp', 'warp1-times', 'warp2-times', 'warp3_times'], loc='lower right')
# plt.savefig('result/data-chart/warp-times-PSNR')
# plt.close()
#
# lines = plt.plot(ID, warp0_ssim, ID, warp1_ssim, ID, warp2_ssim, ID, warp3_ssim)
# plt.setp(lines, marker="o")
# plt.grid(True)
# plt.xlabel('ID')
# plt.ylabel('SSIM')
# plt.legend(['cv_warp', 'warp1-times', 'warp2-times', 'warp3_times'], loc='lower right')
# plt.savefig('result/data-chart/warp-times-SSIM')
# plt.close()