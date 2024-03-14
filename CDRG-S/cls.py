from reg_test_gradient import *
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class pre_train_cls():
    def __init__(self, epochs, batch_num, batch_size):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        #set the model
        self.encoder = encoder()
        self.reg = regression()
        self.cls = cls()
        self.cls_hat = cls_hat()
        self.encoder.load_weights('weights/encoder')
        self.reg.load_weights('weights/reg')

        #set the data path
        self.train2ratio_data, self.train4ratio_data, self.train8ratio_data, self.test2ratio_data, self.test4ratio_data, self.test8ratio_data = self.load_path()
        print(self.train2ratio_data.shape, self.train4ratio_data.shape, self.train8ratio_data.shape, self.test2ratio_data.shape, self.test4ratio_data.shape, self.test8ratio_data.shape)

    def load_path(self):
        path_celeba = '/disk2/bosen/Datasets/celeba_train/'
        train2ratio_data, train4ratio_data, train8ratio_data, test2ratio_data, test4ratio_data, test8ratio_data = [], [], [], [],[], []

        for num, filename in enumerate(os.listdir(path_celeba)):
            if num < 2000:
                image = cv2.imread(path_celeba + filename, 0) / 255
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)
                # _, _, z1 = self.reg(z1)
                # _, _, z2 = self.reg(z2)
                # _, _, z3 = self.reg(z3)
                train2ratio_data.append(tf.reshape(z1, [200]))
                train4ratio_data.append(tf.reshape(z2, [200]))
                train8ratio_data.append(tf.reshape(z3, [200]))

            if 2000 <= num < 2500:
                image = cv2.imread(path_celeba + filename, 0) / 255
                low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)
                # _, _, z1 = self.reg(z1)
                # _, _, z2 = self.reg(z2)
                # _, _, z3 = self.reg(z3)
                test2ratio_data.append(tf.reshape(z1, [200]))
                test4ratio_data.append(tf.reshape(z2, [200]))
                test8ratio_data.append(tf.reshape(z3, [200]))

        train2ratio_data, train4ratio_data, train8ratio_data, test2ratio_data, test4ratio_data, test8ratio_data = np.array(train2ratio_data), np.array(train4ratio_data), np.array(train8ratio_data), np.array(test2ratio_data), np.array(test4ratio_data), np.array(test8ratio_data)
        np.random.shuffle(train2ratio_data)
        np.random.shuffle(train4ratio_data)
        np.random.shuffle(train8ratio_data)

        return train2ratio_data, train4ratio_data, train8ratio_data, test2ratio_data, test4ratio_data, test8ratio_data

    def get_batch_data(self, data, batch_idx, batch_size):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]
        return np.array(train_data)

    def train_step(self, z1, z2, z3, train=True):
        with tf.GradientTape() as tape:
            _, z1_softmax = self.cls(z1)
            _, z2_softmax = self.cls(z2)
            _, z3_softmax = self.cls(z3)
            z1_rec = self.cls_hat(z1_softmax)
            z2_rec = self.cls_hat(z2_softmax)
            z3_rec = self.cls_hat(z3_softmax)

            rec1_loss = tf.reduce_mean(tf.square(z1 - z1_rec))
            rec2_loss = tf.reduce_mean(tf.square(z2 - z2_rec))
            rec3_loss = tf.reduce_mean(tf.square(z3 - z3_rec))

            total_loss = rec1_loss + rec2_loss + rec3_loss

        if train:
            grads = tape.gradient(total_loss, self.cls.trainable_variables + self.cls_hat.trainable_variables)
            tf.keras.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.cls.trainable_variables + self.cls_hat.trainable_variables))
        return rec1_loss, rec2_loss, rec3_loss

    def training(self):
        tr_rec1_loss_epoch = []
        tr_rec2_loss_epoch = []
        tr_rec3_loss_epoch = []
        te_rec1_loss_epoch = []
        te_rec2_loss_epoch = []
        te_rec3_loss_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            tr_rec1_loss_batch = []
            tr_rec2_loss_batch = []
            tr_rec3_loss_batch = []
            te_rec1_loss_batch = []
            te_rec2_loss_batch = []
            te_rec3_loss_batch = []

            for batch in range(self.batch_num):
                tr_z1_batch = self.get_batch_data(self.train2ratio_data, batch, batch_size=self.batch_size)
                tr_z2_batch = self.get_batch_data(self.train4ratio_data, batch, batch_size=self.batch_size)
                tr_z3_batch = self.get_batch_data(self.train8ratio_data, batch, batch_size=self.batch_size)

                tr_rec1_loss, tr_rec2_loss, tr_rec3_loss = self.train_step(tr_z1_batch, tr_z2_batch, tr_z3_batch, train=True)
                tr_rec1_loss_batch.append(tr_rec1_loss)
                tr_rec2_loss_batch.append(tr_rec2_loss)
                tr_rec3_loss_batch.append(tr_rec3_loss)


            for batch in range(int(self.test2ratio_data.shape[0] / 20)):
                te_z1_batch = self.get_batch_data(self.test2ratio_data, batch, batch_size=self.batch_size)
                te_z2_batch = self.get_batch_data(self.test4ratio_data, batch, batch_size=self.batch_size)
                te_z3_batch = self.get_batch_data(self.test8ratio_data, batch, batch_size=self.batch_size)

                te_rec1_loss, te_rec2_loss, te_rec3_loss = self.train_step(te_z1_batch, te_z2_batch, te_z3_batch, train=False)
                te_rec1_loss_batch.append(te_rec1_loss)
                te_rec2_loss_batch.append(te_rec2_loss)
                te_rec3_loss_batch.append(te_rec3_loss)

            tr_rec1_loss_epoch.append(np.mean(tr_rec1_loss_batch))
            tr_rec2_loss_epoch.append(np.mean(tr_rec2_loss_batch))
            tr_rec3_loss_epoch.append(np.mean(tr_rec3_loss_batch))

            te_rec1_loss_epoch.append(np.mean(te_rec1_loss_batch))
            te_rec2_loss_epoch.append(np.mean(te_rec2_loss_batch))
            te_rec3_loss_epoch.append(np.mean(te_rec3_loss_batch))

            print(f'the epoch is {epoch}')
            print(f'the tr rec ratio2 loss is {tr_rec1_loss_epoch[-1]}')
            print(f'the tr rec ratio4 loss is {tr_rec2_loss_epoch[-1]}')
            print(f'the tr rec ratio8 loss is {tr_rec3_loss_epoch[-1]}')

            print(f'the te rec ratio2 loss is {te_rec1_loss_epoch[-1]}')
            print(f'the te rec ratio4 loss is {te_rec2_loss_epoch[-1]}')
            print(f'the te rec ratio8 loss is {te_rec3_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')
            self.cls.save_weights('weights/pretrain_cls_z')
            self.cls_hat.save_weights('weights/pretrain_cls_hat_z')

            plt.plot(tr_rec1_loss_epoch)
            plt.plot(tr_rec2_loss_epoch)
            plt.plot(tr_rec3_loss_epoch)
            plt.legend(['2 ratio', '4 ratio', '8 ratio'], loc='upper right')
            plt.title('Train Rec Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.grid(True)
            plt.savefig('result/cls/pre_train_cls_z_train_rec_loss')
            plt.close()

            plt.plot(te_rec1_loss_epoch)
            plt.plot(te_rec2_loss_epoch)
            plt.plot(te_rec3_loss_epoch)
            plt.legend(['2 ratio', '4 ratio', '8 ratio'], loc='upper right')
            plt.title('Test Rec Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.grid(True)
            plt.savefig('result/cls/pre_train_cls_z_test_rec_loss')
            plt.close()


class train_cls():
    def __init__(self, epochs, batch_num, batch_size, reg_wo):
        #set parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size

        #set the model
        self.encoder = encoder()
        self.reg = regression()
        self.cls = cls()
        self.encoder.load_weights('weights/encoder')
        self.reg.load_weights('weights/reg_dis')
        self.reg_wo = reg_wo

        if self.reg_wo:
            self.cls.load_weights('weights/pretrain_cls_zreg')
        else:
            self.cls.load_weights('weights/pretrain_cls_z')

        #set the data path
        self.train2ratio_latent, self.train4ratio_latent, self.train8ratio_latent, self.val2ratio_latent, self.val4ratio_latent, self.val8ratio_latent, self.train2ratio_label, self.train4ratio_label, self.train8ratio_label, self.val2ratio_label, self.val4ratio_label, self.val8ratio_label = self.load_path()
        print(self.train2ratio_latent.shape, self.train4ratio_latent.shape, self.train8ratio_latent.shape, self.val2ratio_latent.shape, self.val4ratio_latent.shape, self.val8ratio_latent.shape,
              self.train2ratio_label.shape, self.train4ratio_label.shape, self.train8ratio_label.shape, self.val2ratio_label.shape, self.val4ratio_label.shape, self.val8ratio_label.shape)

    def load_path(self):
        path = '/disk2/bosen/Datasets/AR_train/'

        train2ratio_latent, train4ratio_latent, train8ratio_latent, val2ratio_latent, val4ratio_latent, val8ratio_latent = [], [], [], [],[], []
        train2ratio_label, train4ratio_label, train8ratio_label, val2ratio_label, val4ratio_label, val8ratio_label = [], [], [], [],[], []

        for id in os.listdir(path):
            for num, filename in enumerate(os.listdir(path + id)):
                if num < 20:
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                    low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)

                    if self.reg_wo:
                        _, _, z1 = self.reg(z1)
                        _, _, z2 = self.reg(z2)
                        _, _, z3 = self.reg(z3)

                    train2ratio_latent.append(tf.reshape(z1, [200]))
                    train4ratio_latent.append(tf.reshape(z2, [200]))
                    train8ratio_latent.append(tf.reshape(z3, [200]))
                    train2ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                    train4ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                    train8ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))


                if 20 <= num < 100:
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    low1_image = cv2.resize(cv2.resize(image, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(image, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(image, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)

                    if self.reg_wo:
                        _, _, z1 = self.reg(z1)
                        _, _, z2 = self.reg(z2)
                        _, _, z3 = self.reg(z3)

                    val2ratio_latent.append(tf.reshape(z1, [200]))
                    val4ratio_latent.append(tf.reshape(z2, [200]))
                    val8ratio_latent.append(tf.reshape(z3, [200]))
                    val2ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                    val4ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                    val8ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))

        train2ratio_latent, train4ratio_latent, train8ratio_latent, val2ratio_latent, val4ratio_latent, val8ratio_latent = np.array(train2ratio_latent), np.array(train4ratio_latent), np.array(train8ratio_latent), np.array(val2ratio_latent), np.array(val4ratio_latent), np.array(val8ratio_latent)
        train2ratio_label, train4ratio_label, train8ratio_label, val2ratio_label, val4ratio_label, val8ratio_label = np.array(train2ratio_label), np.array(train4ratio_label), np.array(train8ratio_label), np.array(val2ratio_label), np.array(val4ratio_label), np.array(val8ratio_label)

        train2ratio_data = list(zip(train2ratio_latent, train2ratio_label))
        train4ratio_data = list(zip(train4ratio_latent, train4ratio_label))
        train8ratio_data = list(zip(train8ratio_latent, train8ratio_label))
        np.random.shuffle(train2ratio_data)
        np.random.shuffle(train4ratio_data)
        np.random.shuffle(train8ratio_data)
        train2ratio_data = list(zip(*train2ratio_data))
        train4ratio_data = list(zip(*train4ratio_data))
        train8ratio_data = list(zip(*train8ratio_data))

        train2ratio_latent, train2ratio_label = np.array(train2ratio_data[0]), np.array(train2ratio_data[1])
        train4ratio_latent, train4ratio_label = np.array(train4ratio_data[0]), np.array(train4ratio_data[1])
        train8ratio_latent, train8ratio_label = np.array(train8ratio_data[0]), np.array(train8ratio_data[1])

        return train2ratio_latent, train4ratio_latent, train8ratio_latent, val2ratio_latent, val4ratio_latent, val8ratio_latent, train2ratio_label, train4ratio_label, train8ratio_label, val2ratio_label, val4ratio_label, val8ratio_label

    def get_batch_data(self, data, batch_idx, batch_size):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]
        return np.array(train_data)

    def train_step(self, z1_latent, z2_latent, z3_latent, z1_label, z2_label, z3_label, train=True):
        cce = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            _, z1_pred = self.cls(z1_latent)
            _, z2_pred = self.cls(z2_latent)
            _, z3_pred = self.cls(z3_latent)

            ce1_loss = cce(z1_label, z1_pred)
            ce2_loss = cce(z2_label, z2_pred)
            ce3_loss = cce(z3_label, z3_pred)

            acc1 = accuracy_score(np.argmax(z1_label, axis=-1), np.argmax(z1_pred, axis=-1))
            acc2 = accuracy_score(np.argmax(z2_label, axis=-1), np.argmax(z2_pred, axis=-1))
            acc3 = accuracy_score(np.argmax(z3_label, axis=-1), np.argmax(z3_pred, axis=-1))

            total_loss = ce1_loss + ce2_loss + ce3_loss

        if train:
            grads = tape.gradient(total_loss, self.cls.trainable_variables)
            tf.keras.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.cls.trainable_variables))
        return ce1_loss, ce2_loss, ce3_loss, acc1, acc2, acc3

    def training(self):
        tr_ce1_loss_epoch = []
        tr_ce2_loss_epoch = []
        tr_ce3_loss_epoch = []
        tr_acc1_epoch = []
        tr_acc2_epoch = []
        tr_acc3_epoch = []
        val_ce1_loss_epoch = []
        val_ce2_loss_epoch = []
        val_ce3_loss_epoch = []
        val_acc1_epoch = []
        val_acc2_epoch = []
        val_acc3_epoch = []

        for epoch in range(1, self.epochs+1):
            start = time.time()
            tr_ce1_loss_batch = []
            tr_ce2_loss_batch = []
            tr_ce3_loss_batch = []
            tr_acc1_batch = []
            tr_acc2_batch = []
            tr_acc3_batch = []
            val_ce1_loss_batch = []
            val_ce2_loss_batch = []
            val_ce3_loss_batch = []
            val_acc1_batch = []
            val_acc2_batch = []
            val_acc3_batch = []

            for batch in range(int(self.train2ratio_latent.shape[0] / 20)):
                tr_z1_latent_batch, tr_z1_label_batch = self.get_batch_data(self.train2ratio_latent, batch, batch_size=self.batch_size), self.get_batch_data(self.train2ratio_label, batch, batch_size=self.batch_size)
                tr_z2_latent_batch, tr_z2_label_batch = self.get_batch_data(self.train4ratio_latent, batch, batch_size=self.batch_size), self.get_batch_data(self.train4ratio_label, batch, batch_size=self.batch_size)
                tr_z3_latent_batch, tr_z3_label_batch = self.get_batch_data(self.train8ratio_latent, batch, batch_size=self.batch_size), self.get_batch_data(self.train8ratio_label, batch, batch_size=self.batch_size)

                tr_ce1_loss, tr_ce2_loss, tr_ce3_loss, tr_acc1, tr_acc2, tr_acc3 = self.train_step(tr_z1_latent_batch, tr_z2_latent_batch, tr_z3_latent_batch, tr_z1_label_batch, tr_z2_label_batch, tr_z3_label_batch, train=True)
                tr_ce1_loss_batch.append(tr_ce1_loss)
                tr_ce2_loss_batch.append(tr_ce2_loss)
                tr_ce3_loss_batch.append(tr_ce3_loss)
                tr_acc1_batch.append(tr_acc1)
                tr_acc2_batch.append(tr_acc2)
                tr_acc3_batch.append(tr_acc3)


            for batch in range(int(self.val2ratio_latent.shape[0] / 20)):
                val_z1_latent_batch, val_z1_label_batch = self.get_batch_data(self.val2ratio_latent, batch, batch_size=self.batch_size), self.get_batch_data(self.val2ratio_label, batch, batch_size=self.batch_size)
                val_z2_latent_batch, val_z2_label_batch = self.get_batch_data(self.val4ratio_latent, batch, batch_size=self.batch_size), self.get_batch_data(self.val4ratio_label, batch, batch_size=self.batch_size)
                val_z3_latent_batch, val_z3_label_batch = self.get_batch_data(self.val8ratio_latent, batch, batch_size=self.batch_size), self.get_batch_data(self.val8ratio_label, batch, batch_size=self.batch_size)

                val_ce1_loss, val_ce2_loss, val_ce3_loss, val_acc1, val_acc2, val_acc3 = self.train_step(val_z1_latent_batch, val_z2_latent_batch, val_z3_latent_batch, val_z1_label_batch, val_z2_label_batch, val_z3_label_batch, train=False)
                val_ce1_loss_batch.append(val_ce1_loss)
                val_ce2_loss_batch.append(val_ce2_loss)
                val_ce3_loss_batch.append(val_ce3_loss)
                val_acc1_batch.append(val_acc1)
                val_acc2_batch.append(val_acc2)
                val_acc3_batch.append(val_acc3)

            tr_ce1_loss_epoch.append(np.mean(tr_ce1_loss_batch))
            tr_ce2_loss_epoch.append(np.mean(tr_ce2_loss_batch))
            tr_ce3_loss_epoch.append(np.mean(tr_ce3_loss_batch))
            tr_acc1_epoch.append(np.mean(tr_acc1_batch))
            tr_acc2_epoch.append(np.mean(tr_acc2_batch))
            tr_acc3_epoch.append(np.mean(tr_acc3_batch))

            val_ce1_loss_epoch.append(np.mean(val_ce1_loss_batch))
            val_ce2_loss_epoch.append(np.mean(val_ce2_loss_batch))
            val_ce3_loss_epoch.append(np.mean(val_ce3_loss_batch))
            val_acc1_epoch.append(np.mean(val_acc1_batch))
            val_acc2_epoch.append(np.mean(val_acc2_batch))
            val_acc3_epoch.append(np.mean(val_acc3_batch))

            print(f'the epoch is {epoch}')
            print(f'the tr ce ratio2 loss is {tr_ce1_loss_epoch[-1]}')
            print(f'the tr ce ratio4 loss is {tr_ce2_loss_epoch[-1]}')
            print(f'the tr ce ratio8 loss is {tr_ce3_loss_epoch[-1]}')
            print(f'the tr acc ratio2 is {tr_acc1_epoch[-1]}')
            print(f'the tr acc ratio4 is {tr_acc2_epoch[-1]}')
            print(f'the tr acc ratio8 is {tr_acc3_epoch[-1]}')

            print(f'the te ce ratio2 loss is {val_ce1_loss_epoch[-1]}')
            print(f'the te ce ratio4 loss is {val_ce2_loss_epoch[-1]}')
            print(f'the te ce ratio8 loss is {val_ce3_loss_epoch[-1]}')
            print(f'the val acc ratio2 is {val_acc1_epoch[-1]}')
            print(f'the val acc ratio4 is {val_acc2_epoch[-1]}')
            print(f'the val acc ratio8 is {val_acc3_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')

            print('------------------------------------------------')

            if self.reg_wo:
                self.cls.save_weights('weights/cls_zreg_dis')
            else:
                self.cls.save_weights('weights/cls_z')

            plt.plot(tr_ce1_loss_epoch)
            plt.plot(tr_ce2_loss_epoch)
            plt.plot(tr_ce3_loss_epoch)
            plt.legend(['2 ratio', '4 ratio', '8 ratio'], loc='upper right')
            plt.title('Train CE Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.grid(True)
            if self.reg_wo:
                plt.savefig('result/cls/cls_zreg_train_ce_loss')
                plt.close()
            else:
                plt.savefig('result/cls/cls_z_train_ce_loss')
                plt.close()


            plt.plot(tr_acc1_epoch)
            plt.plot(tr_acc2_epoch)
            plt.plot(tr_acc3_epoch)
            plt.legend(['2 ratio', '4 ratio', '8 ratio'], loc='upper right')
            plt.title('Train Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            if self.reg_wo:
                plt.savefig('result/cls/cls_zreg_dis_train_acc')
                plt.close()
            else:
                plt.savefig('result/cls/cls_z_train_acc')
                plt.close()

            plt.plot(val_ce1_loss_epoch)
            plt.plot(val_ce2_loss_epoch)
            plt.plot(val_ce3_loss_epoch)
            plt.legend(['2 ratio', '4 ratio', '8 ratio'], loc='upper right')
            plt.title('Test CE Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.grid(True)
            if self.reg_wo:
                plt.savefig('result/cls/cls_zreg_dis_val_ce_loss')
                plt.close()
            else:
                plt.savefig('result/cls/cls_z_val_ce_loss')
                plt.close()

            plt.plot(val_acc1_epoch)
            plt.plot(val_acc2_epoch)
            plt.plot(val_acc3_epoch)
            plt.legend(['2 ratio', '4 ratio', '8 ratio'], loc='upper right')
            plt.title('Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            if self.reg_wo:
                plt.savefig('result/cls/cls_zreg_dis_val_acc')
                plt.close()
            else:
                plt.savefig('result/cls/cls_z_val_acc')
                plt.close()


class cls_test():
    def __init__(self, reg_wo):
        #set the model
        self.encoder = encoder()
        self.reg = regression()
        self.cls_without_reg = cls()
        self.cls_with_reg = cls()
        self.encoder.load_weights('weights/encoder')
        self.reg.load_weights('weights/reg_dis')
        self.reg_wo = reg_wo

        if self.reg_wo:
            self.cls_with_reg.load_weights('weights/cls_zreg_dis')
        else:
            self.cls_without_reg.load_weights('weights/cls_z')

    def Test_90_id_without_var(self):
        path_val = '/disk2/bosen/Datasets/AR_train/'

        val2ratio_label, val4ratio_label, val8ratio_label = [], [], []
        val2ratio_pred, val4ratio_pred, val8ratio_pred = [], [], []
        for id in os.listdir(path_val):
            for num, filename in enumerate(os.listdir(path_val + id)):
                if 20 <= num < 100:
                    image = cv2.imread(path_val + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                    low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)
                    if self.reg_wo:
                        _, _, z1 = self.reg(z1)
                        _, _, z2 = self.reg(z2)
                        _, _, z3 = self.reg(z3)
                        _, pred1 = self.cls_with_reg(z1)
                        _, pred2 = self.cls_with_reg(z2)
                        _, pred3 = self.cls_with_reg(z3)
                    else:
                        _, pred1 = self.cls_without_reg(z1)
                        _, pred2 = self.cls_without_reg(z2)
                        _, pred3 = self.cls_without_reg(z3)

                    val2ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                    val4ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                    val8ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))

                    val2ratio_pred.append(tf.reshape(pred1, [90]))
                    val4ratio_pred.append(tf.reshape(pred2, [90]))
                    val8ratio_pred.append(tf.reshape(pred3, [90]))

        val2ratio_label, val2ratio_pred = np.array(val2ratio_label), np.array(val2ratio_pred)
        val4ratio_label, val4ratio_pred = np.array(val4ratio_label), np.array(val4ratio_pred)
        val8ratio_label, val8ratio_pred = np.array(val8ratio_label), np.array(val8ratio_pred)

        val2ratio_cm = confusion_matrix(np.argmax(val2ratio_label, axis=-1), np.argmax(val2ratio_pred, axis=-1))
        val2ratio_acc = accuracy_score(np.argmax(val2ratio_label, axis=-1), np.argmax(val2ratio_pred, axis=-1))
        val4ratio_cm = confusion_matrix(np.argmax(val4ratio_label, axis=-1), np.argmax(val4ratio_pred, axis=-1))
        val4ratio_acc = accuracy_score(np.argmax(val4ratio_label, axis=-1), np.argmax(val4ratio_pred, axis=-1))
        val8ratio_cm = confusion_matrix(np.argmax(val8ratio_label, axis=-1), np.argmax(val8ratio_pred, axis=-1))
        val8ratio_acc = accuracy_score(np.argmax(val8ratio_label, axis=-1), np.argmax(val8ratio_pred, axis=-1))

        plt.subplots(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.title(f'2 Ratio Acc {str(val2ratio_acc)[0:5]}')
        plt.imshow(val2ratio_cm, cmap='jet')
        plt.subplot(2, 2, 2)
        plt.title(f'4 Ratio Acc {str(val4ratio_acc)[0:5]}')
        plt.imshow(val4ratio_cm, cmap='jet')
        plt.subplot(2, 2, 3)
        plt.title(f'8 Ratio Acc {str(val8ratio_acc)[0:5]}')
        plt.imshow(val8ratio_cm, cmap='jet')
        plt.show()

    def Test_90_id_with_var(self):
        path_val = '/disk2/bosen/Datasets/AR_aligment_other/'

        val2ratio_label, val4ratio_label, val8ratio_label = [], [], []
        val2ratio_pred, val4ratio_pred, val8ratio_pred = [], [], []
        total_num = 0
        for id in os.listdir(path_val):
            for num, filename in enumerate(os.listdir(path_val + id)):
                print(num)
                total_num += 1
                image = cv2.imread(path_val + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                z1, z2, z3 = self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)

                if self.reg_wo:
                    _, _, z1 = self.reg(z1)
                    _, _, z2 = self.reg(z2)
                    _, _, z3 = self.reg(z3)
                    _, pred1 = self.cls_with_reg(z1)
                    _, pred2 = self.cls_with_reg(z2)
                    _, pred3 = self.cls_with_reg(z3)
                else:
                    _, pred1 = self.cls_without_reg(z1)
                    _, pred2 = self.cls_without_reg(z2)
                    _, pred3 = self.cls_without_reg(z3)

                val2ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                val4ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                val8ratio_label.append(tf.one_hot(int(id[2:]) - 1, 90))

                val2ratio_pred.append(tf.reshape(pred1, [90]))
                val4ratio_pred.append(tf.reshape(pred2, [90]))
                val8ratio_pred.append(tf.reshape(pred3, [90]))

        val2ratio_label, val2ratio_pred = np.array(val2ratio_label), np.array(val2ratio_pred)
        val4ratio_label, val4ratio_pred = np.array(val4ratio_label), np.array(val4ratio_pred)
        val8ratio_label, val8ratio_pred = np.array(val8ratio_label), np.array(val8ratio_pred)

        val2ratio_cm = confusion_matrix(np.argmax(val2ratio_label, axis=-1), np.argmax(val2ratio_pred, axis=-1))
        val2ratio_acc = accuracy_score(np.argmax(val2ratio_label, axis=-1), np.argmax(val2ratio_pred, axis=-1))
        val4ratio_cm = confusion_matrix(np.argmax(val4ratio_label, axis=-1), np.argmax(val4ratio_pred, axis=-1))
        val4ratio_acc = accuracy_score(np.argmax(val4ratio_label, axis=-1), np.argmax(val4ratio_pred, axis=-1))
        val8ratio_cm = confusion_matrix(np.argmax(val8ratio_label, axis=-1), np.argmax(val8ratio_pred, axis=-1))
        val8ratio_acc = accuracy_score(np.argmax(val8ratio_label, axis=-1), np.argmax(val8ratio_pred, axis=-1))
        print(total_num)
        plt.subplots(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.title(f'2 Ratio Acc {str(val2ratio_acc)[0:5]}')
        plt.imshow(val2ratio_cm, cmap='jet')
        plt.subplot(2, 2, 2)
        plt.title(f'4 Ratio Acc {str(val4ratio_acc)[0:5]}')
        plt.imshow(val4ratio_cm, cmap='jet')
        plt.subplot(2, 2, 3)
        plt.title(f'8 Ratio Acc {str(val8ratio_acc)[0:5]}')
        plt.imshow(val8ratio_cm, cmap='jet')
        plt.show()

    def Test_21_id(self):
        path_train = '/disk2/bosen/Datasets/AR_train/'
        path_test = '/disk2/bosen/Datasets/AR_test/'

        database_feature, database_label = [], []
        feature2ratio, feature4ratio, feature8ratio = [], [], []
        feature2ratio_label, feature4ratio_label, feature8ratio_label = [], [], []

        for id in os.listdir(path_train):
            for num, filename in enumerate(os.listdir(path_train + id)):
                if 0 <= num < 20:
                    image = cv2.imread(path_train + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    z0 = self.encoder(image.reshape(1, 64, 64, 1))

                    if self.reg_wo:
                        _, _, z0 = self.reg(z0)
                        feature, _ = self.cls_with_reg(z0)
                    else:
                        feature, _ = self.cls_without_reg(z0)

                    feature = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
                    database_feature.append(tf.reshape(feature, [100]))
                    database_label.append(int(id[2:]) - 1)

        for id in os.listdir(path_test):
            for num, filename in enumerate(os.listdir(path_test + id)):
                if 0 <= num < 20:
                    image = cv2.imread(path_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    z0 = self.encoder(image.reshape(1, 64, 64, 1))

                    if self.reg_wo:
                        _, _, z0 = self.reg(z0)
                        feature, _ = self.cls_with_reg(z0)
                    else:
                        feature, _ = self.cls_without_reg(z0)

                    feature = feature / tf.sqrt(tf.reduce_sum(tf.square(feature)))
                    database_feature.append(tf.reshape(feature, [100]))
                    database_label.append(int(id[2:]) - 1 + 90)

        for id in os.listdir(path_test):
            for num, filename in enumerate(os.listdir(path_test + id)):
                if 20 <= num < 80:
                    image = cv2.imread(path_test + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                    low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC).reshape(1, 64, 64, 1)
                    z0, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image), self.encoder(low2_image), self.encoder(low3_image)

                    if self.reg_wo:
                        _, _, zreg1 = self.reg(z1)
                        _, _, zreg2 = self.reg(z2)
                        _, _, zreg3 = self.reg(z3)
                        feature2, _ = self.cls_with_reg(zreg1)
                        feature4, _ = self.cls_with_reg(zreg2)
                        feature8, _ = self.cls_with_reg(zreg3)
                    else:
                        feature2, _ = self.cls_without_reg(z1)
                        feature4, _ = self.cls_without_reg(z2)
                        feature8, _ = self.cls_without_reg(z3)

                    feature2 = feature2 / tf.sqrt(tf.reduce_sum(tf.square(feature2)))
                    feature4 = feature4 / tf.sqrt(tf.reduce_sum(tf.square(feature4)))
                    feature8 = feature8 / tf.sqrt(tf.reduce_sum(tf.square(feature8)))

                    feature2ratio.append(tf.reshape(feature2, [100]))
                    feature4ratio.append(tf.reshape(feature4, [100]))
                    feature8ratio.append(tf.reshape(feature8, [100]))
                    feature2ratio_label.append(int(id[2:]) + 90 - 1)
                    feature4ratio_label.append(int(id[2:]) + 90 - 1)
                    feature8ratio_label.append(int(id[2:]) + 90 - 1)


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

class retrain_cls():
    def __init__(self):
        # set the model
        self.encoder = encoder()
        self.reg = regression()
        self.generator = generator()
        self.cls= cls()
        self.encoder.load_weights('weights/encoder')
        self.reg.load_weights('weights/reg_x_cls_REG')
        self.generator.load_weights('weights/generator2')

    def prepare_data(self):
        train_path1 = '/disk2/bosen/Datasets/AR_train/'
        train_path2 = '/disk2/bosen/Datasets/AR_aug100_rank6_10_train/'
        test_path = '/disk2/bosen/Datasets/AR_aug100_rank3_5_train/'
        # test_path = '/disk2/bosen/Datasets/AR_aug100_rank6_10_train/'

        train_latent, test_latent = [], []
        train_label, test_label = [], []
        for id in os.listdir(train_path1):
            print(id)
            for num, filename in enumerate(os.listdir(train_path1 + id)):
                if num == 20:
                    break
                image = cv2.imread(train_path1 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                z0, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
                _, _, zreg0 = self.reg(z0)
                _, _, zreg1 = self.reg(z1)
                _, _, zreg2 = self.reg(z2)
                _, _, zreg3 = self.reg(z3)
                train_latent.append(tf.reshape(zreg0, [200]))
                train_latent.append(tf.reshape(zreg1, [200]))
                train_latent.append(tf.reshape(zreg2, [200]))
                train_latent.append(tf.reshape(zreg3, [200]))

                train_label.append(tf.one_hot(int(id[2:])-1, 90))
                train_label.append(tf.one_hot(int(id[2:])-1, 90))
                train_label.append(tf.one_hot(int(id[2:])-1, 90))
                train_label.append(tf.one_hot(int(id[2:])-1, 90))

        print('------load train path2-------')
        # for id in os.listdir(train_path2):
        #     print(id)
        #     for num, filename in enumerate(os.listdir(train_path2 + id)):
        #         if num == 20:
        #             break
        #         image = cv2.imread(train_path2 + id + '/' + filename, 0) / 255
        #         image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        #         blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
        #         low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        #         low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        #         low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
        #         z0, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
        #         _, _, zreg0 = self.reg(z0)
        #         _, _, zreg1 = self.reg(z1)
        #         _, _, zreg2 = self.reg(z2)
        #         _, _, zreg3 = self.reg(z3)
        #         train_latent.append(tf.reshape(zreg0, [200]))
        #         train_latent.append(tf.reshape(zreg1, [200]))
        #         train_latent.append(tf.reshape(zreg2, [200]))
        #         train_latent.append(tf.reshape(zreg3, [200]))
        #
        #         train_label.append(tf.one_hot(int(id[2:]) - 1, 90))
        #         train_label.append(tf.one_hot(int(id[2:]) - 1, 90))
        #         train_label.append(tf.one_hot(int(id[2:]) - 1, 90))
        #         train_label.append(tf.one_hot(int(id[2:]) - 1, 90))

        train_data = list(zip(train_latent, train_label))
        np.random.shuffle(train_data)
        train_data = list(zip(*train_data))
        train_latent, train_label = np.array(train_data[0]), np.array(train_data[1])

        print('------load test path-------')
        for id in os.listdir(test_path):
            print(id)
            for num, filename in enumerate(os.listdir(test_path + id)):
                if num == 20:
                    break
                image = cv2.imread(test_path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                z0, z1, z2, z3 = self.encoder(image.reshape(1, 64, 64, 1)), self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
                _, _, zreg0 = self.reg(z0)
                _, _, zreg1 = self.reg(z1)
                _, _, zreg2 = self.reg(z2)
                _, _, zreg3 = self.reg(z3)
                test_latent.append(tf.reshape(zreg0, [200]))
                test_latent.append(tf.reshape(zreg1, [200]))
                test_latent.append(tf.reshape(zreg2, [200]))
                test_latent.append(tf.reshape(zreg3, [200]))

                test_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                test_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                test_label.append(tf.one_hot(int(id[2:]) - 1, 90))
                test_label.append(tf.one_hot(int(id[2:]) - 1, 90))

        test_latent, test_label = np.array(test_latent), np.array(test_label)
        return train_latent, train_label, test_latent, test_label

    def get_batch_data(self, data, batch_idx, batch_size):
        range_min = batch_idx * batch_size
        range_max = (batch_idx + 1) * batch_size

        if range_max > len(data):
            range_max = len(data)
        index = list(range(range_min, range_max))
        train_data = [data[idx] for idx in index]
        return np.array(train_data)

    def cls_train_step(self, latent, label, lr, train=True):
        cce = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            pred =  self.cls(latent)
            L_ce = cce(label, pred)
            acc = accuracy_score(np.argmax(label, axis=-1), np.argmax(pred, axis=-1))

        if train:
            grads = tape.gradient(L_ce, self.cls.trainable_variables)
            tf.optimizers.Adam(lr).apply_gradients(zip(grads, self.cls.trainable_variables))
        return L_ce, acc

    def main(self, plot=True):
        tr_L_ce_epoch = []
        te_L_ce_epoch = []
        tr_acc_epoch = []
        te_acc_epoch = []

        train_latent, train_label, test_latent, test_label = self.prepare_data()
        print(train_latent.shape, train_label.shape, test_latent.shape, test_label.shape)

        #14400, 7200
        for epoch in range(1, 100):
            start = time.time()
            tr_L_ce_batch = []
            te_L_ce_batch = []
            tr_acc_batch = []
            te_acc_batch = []

            if epoch < 70: learning_rate = 1e-4
            else: learning_rate = 2e-5

            # Train.
            for batch in range(int(train_latent.shape[0]/40)):
                train_latent_batch = self.get_batch_data(train_latent, batch, 40)
                train_label_batch = self.get_batch_data(train_label, batch, 40)

                ce_loss, acc = self.cls_train_step(train_latent_batch, train_label_batch, lr=learning_rate, train=True)
                tr_L_ce_batch.append(ce_loss)
                tr_acc_batch.append(acc)

            #test.
            for batch in range(int(test_latent.shape[0]/40)):
                test_latent_batch = self.get_batch_data(test_latent, batch, 40)
                test_label_batch = self.get_batch_data(test_label, batch, 40)
                ce_loss, acc = self.cls_train_step(test_latent_batch, test_label_batch, lr=learning_rate, train=False)
                te_L_ce_batch.append(ce_loss)
                te_acc_batch.append(acc)

            if plot:
                tr_L_ce_epoch.append(np.mean(tr_L_ce_batch))
                tr_acc_epoch.append(np.mean(tr_acc_batch))
                te_L_ce_epoch.append(np.mean(te_L_ce_batch))
                te_acc_epoch.append(np.mean(te_acc_batch))

                #########################################################
                plt.plot(tr_L_ce_epoch)
                plt.plot(te_L_ce_epoch)
                plt.title('CE Loss')
                plt.xlabel('epoch')
                plt.ylabel('value')
                plt.legend(['train', 'test'], loc='upper right')
                plt.savefig("result/cls_retrain_ce_loss2")
                plt.close()

                #########################################################
                plt.plot(tr_acc_epoch)
                plt.plot(te_acc_epoch)
                plt.title('Accuracy')
                plt.xlabel('epoch')
                plt.ylabel('value')
                plt.legend(['train', 'test'], loc='upper right')
                plt.savefig("result/cls_retrain_acc2")
                plt.close()

                print(f'the epoch is {epoch}')
                print(f'the train ce loss is {tr_L_ce_epoch[-1]}')
                print(f'the train accuracy is {tr_acc_epoch[-1]}')

                print(f'the test ce loss is {te_L_ce_epoch[-1]}')
                print(f'the test accuracy is {te_acc_epoch[-1]}')

                print(f'the spend time is {time.time() - start} second')
                self.cls.save_weights('weights/cls_retrain4')
                print('------------------------------------------------')

    # if train:
    #     # train_latent, train_label, test_latent, test_label = prepare_data()
    #     # print(train_latent.shape, train_label.shape, test_latent.shape, test_label.shape)
    #     #
    #     # cls.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    #     #
    #     # history = cls.fit(train_latent, train_label, epochs=200, batch_size=30, validation_data=(test_latent, test_label), verbose=1)
    #     # cls.save_weights('weights/cls_retrain3')
    #     #
    #     # plt.plot(history.history['loss'], label='Training Loss')
    #     # plt.plot(history.history['val_loss'], label='Validation Loss')
    #     # plt.title('Training and Validation Loss')
    #     # plt.xlabel('Epoch')
    #     # plt.ylabel('Loss')
    #     # plt.legend()
    #     # plt.show()
    #     #
    #     # plt.plot(history.history['accuracy'], label='Training Accuracy')
    #     # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    #     # plt.title('Training and Validation Accuracy')
    #     # plt.xlabel('Epoch')
    #     # plt.ylabel('Accuracy')
    #     # plt.legend()
    #     # plt.show()
    #     main()

    def classifier(self):
        self.cls.load_weights('weights/cls_retrain2')
        test_path = '/disk2/bosen/Datasets/AR_aligment_other/'
        test2ratio, test4ratio, test8ratio = 0, 0, 0
        count = 0
        for id in os.listdir(test_path):
            for num, filename in enumerate(os.listdir(test_path + id)):
                if num == 20:
                    break
                count += 1
                image = cv2.imread(test_path + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                blur_gray = cv2.GaussianBlur(image, (15, 15), 0)
                low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                z1, z2, z3 = self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
                _, _, zreg1 = self.reg(z1)
                _, _, zreg2 = self.reg(z2)
                _, _, zreg3 = self.reg(z3)

                pred2ratio = self.cls(zreg1)
                pred4ratio = self.cls(zreg2)
                pred8ratio = self.cls(zreg3)

                if np.argmax(pred2ratio, axis=-1) == int(id[2:])-1: test2ratio += 1
                if np.argmax(pred4ratio, axis=-1) == int(id[2:])-1: test4ratio += 1
                if np.argmax(pred8ratio, axis=-1) == int(id[2:])-1: test8ratio += 1

        print(f'the 2 ratio accuracy is {test2ratio / count}')
        print(f'the 4 ratio accuracy is {test4ratio / count}')
        print(f'the 8 ratio accuracy is {test8ratio / count}')

    def knn_high_images(self):
        train_path1 = '/disk2/bosen/Datasets/AR_train/'
        train_path2 = '/disk2/bosen/Datasets/AR_test/'

        test_path1 = '/disk2/bosen/Datasets/AR_test/'
        test_path2 = '/disk2/bosen/Datasets/AR_aligment_other/'
        test_path3 = "/disk2/bosen/Datasets/AR_aligment_final_without_Occ_and_Expression/"

        database_high_images, database_label = [], []
        for id in os.listdir(train_path1):
            for num, filename in enumerate(os.listdir(train_path1+id)):
                if num == 10:
                    break
                image = cv2.imread(train_path1 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                database_high_images.append(image)
                database_label.append(int(id[2:]))

        for id in os.listdir(train_path2):
            for num, filename in enumerate(os.listdir(train_path2+id)):
                if num == 10:
                    break
                image = cv2.imread(train_path2 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                database_high_images.append(image)
                database_label.append(int(id[2:])+90)

        database_high_images, database_label = np.array(database_high_images), np.array(database_label)
        database_high_images = database_high_images.reshape(database_high_images.shape[0], -1)
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(database_high_images, database_label)
        for i in database_label:
            print(i)
        print(database_high_images.shape)


        def test(knn_classifier, test_path):
            if test_path == 'db1': path = test_path1
            elif test_path == 'db2': path = test_path2
            elif test_path == 'db3': path = test_path3


            db1_test = [[]for i in range(3)]
            db1_label = []

            for id in os.listdir(path):
                for num, filename in enumerate(os.listdir(path+id)):
                    if test_path == 'db1':
                        if num < 10:
                            continue
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                    low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    z1, z2, z3 = self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
                    _, _, zreg1 = self.reg(z1)
                    _, _, zreg2 = self.reg(z2)
                    _, _, zreg3 = self.reg(z3)
                    syn1 = self.generator(zreg1)
                    syn2 = self.generator(zreg2)
                    syn3 = self.generator(zreg3)

                    db1_test[0].append(tf.reshape(syn1, [64, 64]))
                    db1_test[1].append(tf.reshape(syn2, [64, 64]))
                    db1_test[2].append(tf.reshape(syn3, [64, 64]))
                    if test_path == 'db1':
                        db1_label.append(int(id[2:])+90)
                    else:
                        db1_label.append(int(id[2:]))

            db1_test = np.array(db1_test)
            ratio2_test = db1_test[0].reshape(db1_test[0].shape[0], -1)
            ratio4_test = db1_test[1].reshape(db1_test[1].shape[0], -1)
            ratio8_test = db1_test[2].reshape(db1_test[2].shape[0], -1)

            db1_label = np.array(db1_label)
            print(db1_label)
            print(ratio2_test.shape)
            print(ratio4_test.shape)
            print(ratio8_test.shape)

            pred1 = knn_classifier.predict(ratio2_test)
            pred2 = knn_classifier.predict(ratio4_test)
            pred3 = knn_classifier.predict(ratio8_test)

            feature2ratio_cm = confusion_matrix(db1_label, pred1)
            feature2ratio_acc = accuracy_score(db1_label, pred1)
            feature4ratio_cm = confusion_matrix(db1_label, pred2)
            feature4ratio_acc = accuracy_score(db1_label, pred2)
            feature8ratio_cm = confusion_matrix(db1_label, pred3)
            feature8ratio_acc = accuracy_score(db1_label, pred3)

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
            plt.savefig(f'result/cls/knn_high_images_{test_path}')
            plt.close()

        test(knn_classifier, test_path='db1')
        test(knn_classifier, test_path='db2')
        test(knn_classifier, test_path='db3')


    def knn_latent_code(self, latent='z'):
        train_path1 = '/disk2/bosen/Datasets/AR_train/'
        train_path2 = '/disk2/bosen/Datasets/AR_test/'

        test_path1 = '/disk2/bosen/Datasets/AR_test/'
        test_path2 = '/disk2/bosen/Datasets/AR_aligment_other/'
        test_path3 = "/disk2/bosen/Datasets/AR_aligment_final_without_Occ_and_Expression/"

        database_latent, database_label = [], []
        for id in os.listdir(train_path1):
            for num, filename in enumerate(os.listdir(train_path1 + id)):
                if num == 10:
                    break
                image = cv2.imread(train_path1 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                z = self.encoder(image.reshape(1, 64, 64, 1))
                if latent == 'z':
                    z = z / tf.sqrt(tf.reduce_sum(tf.square(z)))
                    database_latent.append(tf.reshape(z, [200]))
                else:
                    _, _, zreg = self.reg(z)
                    zreg = zreg / tf.sqrt(tf.reduce_sum(tf.square(zreg)))
                    database_latent.append(tf.reshape(zreg, [200]))
                database_label.append(int(id[2:]))

        for id in os.listdir(train_path2):
            for num, filename in enumerate(os.listdir(train_path2 + id)):
                if num == 10:
                    break
                image = cv2.imread(train_path2 + id + '/' + filename, 0) / 255
                image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                z = self.encoder(image.reshape(1, 64, 64, 1))
                if latent == 'z':
                    z = z / tf.sqrt(tf.reduce_sum(tf.square(z)))
                    database_latent.append(tf.reshape(z, [200]))
                else:
                    _, _, zreg = self.reg(z)
                    zreg = zreg / tf.sqrt(tf.reduce_sum(tf.square(zreg)))
                    database_latent.append(tf.reshape(zreg, [200]))
                database_label.append(int(id[2:])+90)

        database_latent, database_label = np.array(database_latent), np.array(database_label)
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(database_latent, database_label)
        print(database_latent.shape)

        def test(test_path):
            if test_path == 'db1': path = test_path1
            elif test_path == 'db2': path = test_path2
            elif test_path == 'db3': path = test_path3

            db1_test = [[] for i in range(3)]
            db1_label = []

            for id in os.listdir(path):
                for num, filename in enumerate(os.listdir(path + id)):
                    image = cv2.imread(path + id + '/' + filename, 0) / 255
                    image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)

                    low1_image = cv2.resize(cv2.resize(blur_gray, (32, 32), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low2_image = cv2.resize(cv2.resize(blur_gray, (16, 16), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)
                    low3_image = cv2.resize(cv2.resize(blur_gray, (8, 8), cv2.INTER_CUBIC), (64, 64), cv2.INTER_CUBIC)

                    z1, z2, z3 = self.encoder(low1_image.reshape(1, 64, 64, 1)), self.encoder(low2_image.reshape(1, 64, 64, 1)), self.encoder(low3_image.reshape(1, 64, 64, 1))
                    if latent == 'z':
                        z1 = z1 / tf.sqrt(tf.reduce_sum(tf.square(z1)))
                        z2 = z2 / tf.sqrt(tf.reduce_sum(tf.square(z2)))
                        z3 = z3 / tf.sqrt(tf.reduce_sum(tf.square(z3)))
                        db1_test[0].append(tf.reshape(z1, [200]))
                        db1_test[1].append(tf.reshape(z2, [200]))
                        db1_test[2].append(tf.reshape(z3, [200]))
                    else:
                        _, _, zreg1 = self.reg(z1)
                        _, _, zreg2 = self.reg(z2)
                        _, _, zreg3 = self.reg(z3)
                        zreg1 = zreg1 / tf.sqrt(tf.reduce_sum(tf.square(zreg1)))
                        zreg2 = zreg2 / tf.sqrt(tf.reduce_sum(tf.square(zreg2)))
                        zreg3 = zreg3 / tf.sqrt(tf.reduce_sum(tf.square(zreg3)))
                        db1_test[0].append(tf.reshape(zreg1, [200]))
                        db1_test[1].append(tf.reshape(zreg2, [200]))
                        db1_test[2].append(tf.reshape(zreg3, [200]))

                    if test_path == 'db1':
                        db1_label.append(int(id[2:])+90)
                    else:
                        db1_label.append(int(id[2:]))

            db1_test = np.array(db1_test)
            ratio2_test = db1_test[0].reshape(db1_test[0].shape[0], -1)
            ratio4_test = db1_test[1].reshape(db1_test[1].shape[0], -1)
            ratio8_test = db1_test[2].reshape(db1_test[2].shape[0], -1)

            db1_label = np.array(db1_label)
            print(ratio2_test.shape)
            print(ratio4_test.shape)
            print(ratio8_test.shape)

            pred1 = knn_classifier.predict(ratio2_test)
            pred2 = knn_classifier.predict(ratio4_test)
            pred3 = knn_classifier.predict(ratio8_test)

            feature2ratio_cm = confusion_matrix(db1_label, pred1)
            feature2ratio_acc = accuracy_score(db1_label, pred1)
            feature4ratio_cm = confusion_matrix(db1_label, pred2)
            feature4ratio_acc = accuracy_score(db1_label, pred2)
            feature8ratio_cm = confusion_matrix(db1_label, pred3)
            feature8ratio_acc = accuracy_score(db1_label, pred3)

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
            plt.savefig(f'result/cls/knn_latent_{latent}_{test_path}')
            plt.close()

        test(test_path='db1')
        test(test_path='db2')
        test(test_path='db3')


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    cls = retrain_cls()
    # cls.main(plot=True)
    # cls.classifier()
    # cls.knn_high_images()
    cls.knn_latent_code(latent='z')
    cls.knn_latent_code(latent='zreg')
