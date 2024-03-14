import time, datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
import options
import cv2
import network
import load_data
import warp
import util

print(util.toYellow("======================================================="))
print(util.toYellow("main.py (training on MNIST)"))
print(util.toYellow("======================================================="))

class Project():
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(self.current_time)
        self.opt = options.set(training=True)

        # self.train_datasets, self.val_datasets, self.test_datasets = load_data.load_mnist(batchSize=self.opt.batchSize)
        self.train_path, self.test_path = load_data.load_data_path()
        self.avg_face = np.load("/home/ruiteng/Desktop/dr_reg/teacher_student_siamese/augment/avg_face.npy")
        self.avg_face = self.avg_face.reshape(128, 128)
        O = np.zeros((128))
        I = np.ones((128))
        self.mask_for_avg = np.concatenate((np.stack([O for i in range(64)]), np.stack([I for i in range(64)])), axis=0)
        self.mask_for_affine = np.concatenate((np.stack([I for i in range(64)]), np.stack([O for i in range(64)])), axis=0)
        self.avg_face = self.avg_face * self.mask_for_avg
        # -------------------- Create model --------------------
        self.IC_STN = network.IC_STN(self.opt.warpDim)
        self.Enc = network.Enc()
        self.Cls = network.ID_Cls()

        self.op_IC_STN = tf.optimizers.Adam()
        self.op_CLS = tf.optimizers.SGD()

        # -------------------- Create writer --------------------
        self.IC_STN_checkpoint = tf.train.Checkpoint(model=self.IC_STN)
        self.Enc_checkpoint = tf.train.Checkpoint(model=self.Enc)
        self.Cls_checkpoint = tf.train.Checkpoint(model=self.Cls)

        # self.IC_STN_checkpoint.restore('./IC_STN_on_Mnist/checkpoints/ckpt-60')
        self.Enc_checkpoint.restore('CLS_on_AR_occ_and_nonocc/checkpoints/Encoder/ckpt-50')
        self.Cls_checkpoint.restore('CLS_on_AR_occ_and_nonocc/checkpoints/Classifier/ckpt-50')


        test_sample_path = self.test_path[0: 5]
        self.testing_example = []
        for test in test_sample_path:
            image = cv2.imread(test) / 255
            self.testing_example.append(image)
        self.testing_example = np.array(self.testing_example)

    def calculate_loss_acc(self, y_true, y_pred):
        # labelOnehot = tf.one_hot(y_true, self.opt.labelN)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        acc = accuracy_score(y_true, y_pred)
        return loss, acc

    def train_step(self, p, images, label):
        with tf.GradientTape() as GP_tape, tf.GradientTape() as C_tape:
            imageWarpAll = []
            for l in range(self.opt.warpN):
                pMtrx = warp.vec2mtrx(self.opt, p)
                imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)

                imageWarpAll.append(imageWarp)

                feat = self.IC_STN(imageWarp)
                dp = feat
                p = warp.compose(self.opt, p, dp)
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, test, _ = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            imageWarp = imageWarpAll[-1]
            imageWarp = tf.image.rgb_to_grayscale(imageWarp)


            predictions = self.Enc(imageWarp)
            # predictions = self.Cls(predictions)
            loss = test
            # loss, acc = self.calculate_loss_acc(label, predictions)
        gradients_of_GP = GP_tape.gradient(loss, self.IC_STN.trainable_variables)
        gradients_of_c = C_tape.gradient(loss, self.Enc.trainable_variables + self.Cls.trainable_variables)
        self.op_IC_STN.apply_gradients(zip(gradients_of_GP, self.IC_STN.trainable_variables))
        print('grads')
        print(gradients_of_GP)
        self.op_CLS.apply_gradients(zip(gradients_of_c, self.Enc.trainable_variables + self.Cls.trainable_variables))
        return loss, acc

    def test_step(self, p, images, label=None):
        imageWarpAll = []
        for l in range(self.opt.warpN):
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            feat = self.IC_STN(imageWarp)
            dp = feat
            p = warp.compose(self.opt, p, dp)
        pMtrx = warp.vec2mtrx(self.opt, p)
        imageWarp, XfloorInt, YfloorInt = warp.transformImage(self.opt, images, pMtrx)
        imageWarpAll.append(imageWarp)
        imageWarp = imageWarpAll[-1]
        imageWarp = tf.image.rgb_to_grayscale(imageWarp)
        predictions = self.Enc(imageWarp)
        predictions = self.Cls(predictions)
        loss, acc = 0, 0
        # if label != None:
        if label is not None:
            loss, acc = self.calculate_loss_acc(label, predictions)
        return loss, acc, imageWarp, XfloorInt, YfloorInt

    def plot_Ori_region(self, Ori_image, imageWarp, X_coordinate, Y_coordinate, epoch, pad_width=60):
        epoch = epoch + 1
        print('plot')
        util.mkdir(f'./testing_plot')
        util.mkdir(f'./testing_plot/{self.current_time}')
        for i in range(Ori_image.shape[0]):
            util.mkdir(f'./testing_plot/{self.current_time}/test{i + 1}')

            X_coord = X_coordinate[i + 1] + pad_width
            Y_coord = Y_coordinate[i + 1] + pad_width
            X_coord = tf.reshape(X_coord, [128*128])
            Y_coord = tf.reshape(Y_coord, [128*128])

            x1, x2, x3, x4 = X_coord[0], X_coord[127], X_coord[128*128-1-127], X_coord[128*128-1]
            y1, y2, y3, y4 = Y_coord[0], Y_coord[127], Y_coord[128*128-1-127], Y_coord[128*128-1]

            # ori = Ori_image[i].numpy().reshape((128, 128)) * 255.
            # print(ori)
            Ori_img = tf.image.rgb_to_grayscale(Ori_image[i])
            ori = tf.reshape(Ori_img, [192, 256]) * 255
            padding = np.pad(ori, pad_width, 'constant', constant_values=255)
            img = tf.reshape(tf.cast(padding, tf.uint8), [192 + pad_width * 2, 256 + pad_width * 2, 1])

            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.plot([x1, x2], [y1, y2], color='red')
            plt.plot([x2, x4], [y2, y4], color='red')
            plt.plot([x3, x4], [y3, y4], color='red')
            plt.plot([x3, x1], [y3, y1], color='red')
            plt.savefig(f'./testing_plot/{self.current_time}/test{i + 1}/epoch{epoch}_ori')
            plt.close()

            plt.figure()
            img = tf.cast(tf.reshape(imageWarp[i] * 255., [128, 128]), tf.uint8)
            plt.imshow(img, cmap='gray')
            plt.savefig(f'./testing_plot/{self.current_time}/test{i + 1}/epoch{epoch}_warp')
            plt.close()

    def run(self):
        L_tr_epoch, L_te_epoch = [], []
        A_tr_epoch, A_te_epoch = [], []
        for epoch in range(self.opt.epochs):
            start = time.time()
            # ---------training---------
            L_tr_batch, L_te_batch = [], []
            A_tr_batch, A_te_batch = [], []
            for batch in range(self.opt.train_batchnum):
                batch_image, batch_label = load_data.load_batch_image(self.train_path, batch, self.opt.batchSize)
                Init_p = load_data.genPerturbations(self.opt)
                loss_tr, acc_tr = self.train_step(Init_p, batch_image, batch_label)
                L_tr_batch.append(loss_tr)
                A_tr_batch.append(acc_tr)

            for batch in range(self.opt.test_batchnum):
                batch_image, batch_label = load_data.load_batch_image(self.test_path, batch, self.opt.batchSize)
                Init_p = load_data.genPerturbations(self.opt)
                loss_te, acc_te, _, _, _ = self.test_step(Init_p, batch_image, batch_label)
                L_te_batch.append(loss_te)
                A_te_batch.append(acc_te)

            L_tr_epoch.append(np.mean(L_tr_batch))
            L_te_epoch.append(np.mean(L_te_batch))
            A_tr_epoch.append(np.mean(A_tr_batch))
            A_te_epoch.append(np.mean(A_te_batch))

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            template = 'Train Loss: {}, Train Accuracy: {}, test Loss: {}, test Accuracy: {}'
            print(template.format(tf.reduce_mean(L_te_batch).numpy(), tf.reduce_mean(A_tr_batch).numpy(),
                                  tf.reduce_mean(L_te_batch).numpy(), tf.reduce_mean(A_te_batch).numpy()))

            # # update learning rate
            # # lrGP = self.opt.lrGP * (self.opt.lrGPdecay ** tf.cast((iter // self.opt.lrGPstep), tf.float32))
            # # lrC = self.opt.lrC * (self.opt.lrCdecay ** tf.cast((iter // self.opt.lrCstep), tf.float32))


            # decay = tf.cast(self.opt.lrGP / self.opt.epochs, tf.float32)
            # lrGP = tf.cast(self.opt.lrGP * 1. / (1. + decay * epoch), tf.float32)
            # lrC = tf.cast(self.opt.lrC * 1. / (1. + decay * epoch), tf.float32)
            lrGP = self.opt.lrGP
            lrC = self.opt.lrC
            print(f'epoch: {epoch + 1}, lrGP: {lrGP}, lrC: {lrC}')
            print('----------------------------------------------------------------')

            # self.op_IC_STN = tf.optimizers.SGD(lrGP, momentum=0.8)
            # self.op_CNN = tf.optimizers.SGD(lrC, momentum=0.8)


            if (epoch + 1) % 2 == 0:
                # self.IC_STN_manager.save(epoch + 1)
                # ---------plot 前 testing 5筆 warp狀況---------
                Init_p = load_data.genPerturbations(self.opt)
                _, _, imageWarp, x_cor, y_cor = self.test_step(Init_p, self.testing_example)
                self.plot_Ori_region(self.testing_example, imageWarp, x_cor, y_cor, epoch, pad_width=8)



if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    project = Project()
    project.run()