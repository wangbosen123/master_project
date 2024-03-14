import time, datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
import copy
import options
import network
import load_data
import warp
import util

print(util.toYellow("======================================================="))
print(util.toYellow("main.py (training on AR Face)"))
print(util.toYellow("======================================================="))

class Project():

    def __init__(self):

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(self.current_time)
        self.opt = options.set(training=True)

        # -------------------- Create model --------------------
        self.IC_STN = network.IC_STN(self.opt.warpDim)
        self.Enc = network.Enc()
        self.Cls = network.ID_Cls()

        # self.op_IC_STN = tf.optimizers.SGD(self.opt.lrGP)
        # self.op_CNN = tf.optimizers.SGD(self.opt.lrC)
        self.op_IC_STN = tf.optimizers.Adam()
        self.op_CLS = tf.optimizers.SGD()

        # -------------------- Create writer --------------------

        self.train_writer = tf.summary.create_file_writer(f'./{self.current_time}/logs/train')
        self.valid_writer = tf.summary.create_file_writer(f'./{self.current_time}/logs/valid')

        self.IC_STN_checkpoint = tf.train.Checkpoint(model=self.IC_STN)
        self.Enc_checkpoint = tf.train.Checkpoint(model=self.Enc)
        self.Cls_checkpoint = tf.train.Checkpoint(model=self.Cls)

        self.IC_STN_checkpoint.restore('./IC_STN_on_Mnist/checkpoints/ckpt-60')
        self.Enc_checkpoint.restore('./CLS_on_AR_occ_and_nonocc/checkpoints/Encoder/ckpt-50')
        self.Cls_checkpoint.restore('./CLS_on_AR_occ_and_nonocc/checkpoints/Classifier/ckpt-50')

        self.IC_STN_manager = tf.train.CheckpointManager(self.IC_STN_checkpoint, directory=f'./{self.current_time}/checkpoints/IC_STN/', max_to_keep=30)
        self.Enc_manager = tf.train.CheckpointManager(self.Enc_checkpoint, directory=f'./{self.current_time}/checkpoints/Enc/', max_to_keep=30)
        self.Cls_manager = tf.train.CheckpointManager(self.Cls_checkpoint, directory=f'./{self.current_time}/checkpoints/Cls/', max_to_keep=30)

    def calculate_loss_acc(self, y_true, y_pred):
        labelOnehot = tf.one_hot(y_true, self.opt.labelN)
        loss = tf.keras.losses.categorical_crossentropy(y_pred, labelOnehot)
        loss = tf.reduce_mean(loss)

        y_pred = tf.argmax(y_pred, axis=-1)
        acc = accuracy_score(y_true, y_pred)
        return loss, acc

    def train_step(self, p, images, label, is_training=False):
        with tf.GradientTape() as GP_tape, tf.GradientTape() as C_tape:
            imageWarpAll = []
            for l in range(self.opt.warpN):
                pMtrx = warp.vec2mtrx(self.opt, p)
                imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)
                imageWarpAll.append(imageWarp)
                feat = self.IC_STN.call(imageWarp)
                dp = feat
                p = warp.compose(self.opt, p, dp)
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            imageWarp = imageWarpAll[-1]
            predictions = self.Cls.call(self.Enc.call(imageWarp))
            predictions = tf.nn.softmax(predictions, axis=-1)
            loss, acc = self.calculate_loss_acc(label, predictions)
        if is_training == True:
            gradients_of_GP = GP_tape.gradient(loss, self.IC_STN.trainable_variables)
            gradients_of_C = C_tape.gradient(loss, self.Enc.trainable_variables + self.Cls.trainable_variables)
            self.op_IC_STN.apply_gradients(zip(gradients_of_GP, self.IC_STN.trainable_variables))
            self.op_CLS.apply_gradients(zip(gradients_of_C, self.Enc.trainable_variables + self.Cls.trainable_variables))
        return loss, acc

    def test_step(self, p, images, label):
        imageWarpAll = []
        for l in range(self.opt.warpN):
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, XfloorInt, YfloorInt = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            feat = self.IC_STN.call(imageWarp)
            dp = feat
            p = warp.compose(self.opt, p, dp)
        pMtrx = warp.vec2mtrx(self.opt, p)
        imageWarp, XfloorInt, YfloorInt = warp.transformImage(self.opt, images, pMtrx)
        imageWarpAll.append(imageWarp)
        imageWarp = imageWarpAll[-1]
        predictions = self.Cls.call(self.Enc.call(imageWarp))
        predictions = tf.nn.softmax(predictions, axis=-1)
        loss, acc = 0, 0
        if label != None:
            loss, acc = self.calculate_loss_acc(label, predictions)
        return loss, acc, imageWarp, XfloorInt, YfloorInt

    def run(self):

        def training(epoch, is_training):
            start = time.time()
            # ---------training---------
            L_tr = []
            A_tr = []
            iter = 0
            for i in range(len(x_train)):
                image_path_batch, label_batch = x_train[i], np.array(y_train[i])
                image_batch = load_data.read_img(image_path_batch)
                Init_p = load_data.genPerturbations(self.opt)
                loss_tr, acc_tr = self.train_step(Init_p, image_batch, label_batch, is_training)
                L_tr.append(loss_tr)
                A_tr.append(acc_tr)

                print(f'\rtrain- {iter}/{len(x_train)}, '
                      f'tr_id_loss:{loss_tr}, '
                      f'tr_id_acc:{acc_tr}, ', end="")
                iter += 1

            # ---------validation---------
            L_val = []
            A_val = []
            for i in range(len(x_val)):
                image_path_batch, label_batch = x_val[i], np.array(y_val[i])
                image_batch = load_data.read_img(image_path_batch)
                Init_p = load_data.genPerturbations(self.opt)
                loss_val, acc_val, _, _, _ = self.test_step(Init_p, image_batch, label_batch)
                L_val.append(loss_val)
                A_val.append(acc_val)

            with self.train_writer.as_default():
                tf.summary.scalar('loss(epoch)', tf.reduce_mean(L_tr), step=epoch + 1)
                tf.summary.scalar('acc(epoch)', tf.reduce_mean(A_tr), step=epoch + 1)

            with self.valid_writer.as_default():
                tf.summary.scalar('loss(epoch)', tf.reduce_mean(L_val), step=epoch + 1)
                tf.summary.scalar('acc(epoch)', tf.reduce_mean(A_val), step=epoch + 1)

            print()
            print('============ Time for epoch {} is {} sec ============'.format(epoch + 1, time.time() - start))
            print('Train Loss: %.8f' % tf.reduce_mean(L_tr).numpy(), 'Train Accuracy: %.8f' %  tf.reduce_mean(A_tr).numpy())
            print('Val Loss: %.8f' % tf.reduce_mean(L_val).numpy(), 'Val Accuracy: %.8f' % tf.reduce_mean(A_val).numpy())

            # # update learning rate
            # # lrGP = self.opt.lrGP * (self.opt.lrGPdecay ** tf.cast((iter // self.opt.lrGPstep), tf.float32))
            # # lrC = self.opt.lrC * (self.opt.lrCdecay ** tf.cast((iter // self.opt.lrCstep), tf.float32))

            # decay = tf.cast(self.opt.lrGP / self.opt.epochs, tf.float32)
            # lrGP = tf.cast(self.opt.lrGP * 1. / (1. + decay * epoch), tf.float32)
            # lrC = tf.cast(self.opt.lrC * 1. / (1. + decay * epoch), tf.float32)
            # print(f'epoch: {epoch}, lrGP: {lrGP}, lrC: {lrC}')
            # self.op_IC_STN = tf.optimizers.SGD(lrGP, momentum=0.8)
            # self.op_CNN = tf.optimizers.SGD(lrC, momentum=0.8)
            # with self.train_writer.as_default():
            #     tf.summary.scalar('lrGP(epoch)', lrGP, step=epoch)
            #     tf.summary.scalar('lrC(epoch)', lrC, step=epoch)

            if (epoch + 1) % 10 == 0:
                self.IC_STN_manager.save(epoch + 1)
                self.Enc_manager.save(epoch + 1)
                self.Cls_manager.save(epoch + 1)

                # # ---------plot 5筆 warp狀況---------
                # Init_p = load_data.genPerturbations(self.opt)
                # _, _, imageWarp, x_cor, y_cor = self.test_step(Init_p, testing_example)
                # self.plot_Ori_region(self.testing_example, imageWarp, x_cor, y_cor, epoch, pad_width=8)

        def balabce_batch(img_path_list, N=3):
            # training data output dir
            # img_path_list=
            # [[ID1 100p],
            #  [ID2 100p],
            #  ...
            #  [ID90 100p]]
            ID_num = len(img_path_list)

            num_of_each_ID = [len(img_path_list[i]) for i in range(ID_num)]
            min_num = np.min(num_of_each_ID)
            # print('min_num', min_num)
            batch_num = int(min_num / N)
            # print('batch_num', batch_num)

            Path = []
            Label = []
            L = copy.deepcopy(img_path_list)
            # print('L', L)
            for b in range(batch_num):
                path = []
                label = []
                for id in range(ID_num):
                    random_pick_num = np.random.choice(np.arange(len(L[id])), N, replace=False)
                    # print('random_pick_num', random_pick_num)
                    random_pick_num = sorted(random_pick_num, reverse=True)
                    # print('sorted_random_pick_num', random_pick_num)
                    for n in range(N):
                        # print(random_pick_num[n])
                        path.append(L[id][random_pick_num[n]])
                        label.append(id)
                        L[id].pop(random_pick_num[n])
                Path.append(path)
                Label.append(label)
            return Path, Label

        # get path
        train_datasets = load_data.get_AR_aug_data_path_by_ID_order(aug_path='/home/fagc2267/traindata/AR_aug100_rank1_2_train')
        val_datasets = load_data.get_AR_aug_data_path_by_ID_order(aug_path='')
        # test_datasets = load_data.get_AR_aug_data_path_by_ID_order(aug_path='')

        x_train, y_train = balabce_batch(train_datasets, N=1) # 一個batch內每個ID隨機取N個
        x_val, y_val = balabce_batch(val_datasets, N=1)
        # plot example
        self.testing_example = load_data.read_img(x_val[0][0:5])
        self.testing_example_y = y_val[0][0:5]
        training(epoch=-1, is_training=False)
        for epoch in range(self.opt.epochs):
            x_train, y_train = balabce_batch(train_datasets, N=1)
            x_val, y_val = balabce_batch(val_datasets, N=1)
            training(epoch=epoch, is_training=True)

        # ---------testing--------- 以下再改成你們要丟的方式
        # A_te = []
        # for image_batch, label_batch in test_datasets:
        #     Init_p = load_data.genPerturbations(self.opt)
        #     _, acc_te, _, _, _ = self.test_step(Init_p, image_batch, label_batch)
        #     A_te.append(acc_te)
        # print(f'testing acc:{tf.reduce_mean(A_te).numpy()}')

    # -------------------------------------------------------------------
    def test_warpNprocess(self, p, images):
        imageWarpAll = []
        x = []
        y = []
        for l in range(self.opt.warpN):
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, XfloorInt, YfloorInt = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            feat = self.IC_STN.call(imageWarp)
            dp = feat
            p = warp.compose(self.opt, p, dp)
        pMtrx = warp.vec2mtrx(self.opt, p)
        imageWarp, XfloorInt, YfloorInt = warp.transformImage(self.opt, images, pMtrx)
        imageWarpAll.append(imageWarp)
        x.append(XfloorInt)
        y.append(YfloorInt)
        return imageWarpAll, x, y

    # 需改: 原本 Mnist 是28x28=784改成 AR 128x128 (看在大張影像上的框框位置，與warp後的小張影像)
    def plot_Ori_region(self, Ori_image, imageWarp, X_coordinate, Y_coordinate, warpn, pad_width=8):

        # # ---------plot 前 testing 5筆 warp狀況--------- 使用方式
        # Init_p = load_data.genPerturbations(self.opt)
        # imageWarpAll, _x, _y = self.test_warpNprocess(Init_p, testing_example)
        # for i in range(self.opt.warpN+1):
        #     self.plot_Ori_region(self.testing_example, imageWarpAll[i], _x[i], _y[i], i, pad_width=8)

        warpn = warpn + 1
        print('plot')
        util.mkdir(f'./testing_plot')
        util.mkdir(f'./{self.current_time}/testing_plot')
        for i in range(Ori_image.shape[0]):
            util.mkdir(f'./{self.current_time}/testing_plot/test{i + 1}')

            X_coord = X_coordinate[i + 1] + pad_width
            Y_coord = Y_coordinate[i + 1] + pad_width

            X_coord = tf.reshape(X_coord, [784])
            Y_coord = tf.reshape(Y_coord, [784])

            x1, x2, x3, x4 = X_coord[0], X_coord[27], X_coord[756], X_coord[783]
            y1, y2, y3, y4 = Y_coord[0], Y_coord[27], Y_coord[756], Y_coord[783]

            ori = Ori_image[i].numpy().reshape((28, 28)) * 255.
            padding = np.pad(ori, pad_width, 'constant', constant_values=255)
            img = tf.reshape(tf.cast(padding, tf.uint8), [28 + pad_width * 2, 28 + pad_width * 2, 1])

            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.plot([x1, x2], [y1, y2], color='red')
            plt.plot([x2, x4], [y2, y4], color='red')
            plt.plot([x3, x4], [y3, y4], color='red')
            plt.plot([x3, x1], [y3, y1], color='red')
            plt.savefig(f'./{self.current_time}/testing_plot//test{i + 1}/warpn{warpn}_ori')
            plt.close('all')

            plt.figure()
            img = tf.cast(tf.reshape(imageWarp[i] * 255., [28, 28]), tf.uint8)
            plt.imshow(img, cmap='gray')
            plt.savefig(f'./{self.current_time}/testing_plot/test{i + 1}/warpn{warpn}_warp')
            plt.close('all')

if __name__ == '__main__':
    project = Project()
    project.run()