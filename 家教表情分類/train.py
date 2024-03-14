import tensorflow as tf
import numpy as np
import time
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from load_data import *
import statistics


def overall_model():
    # model = load_model("vgg16_recognize_emotion_student.h5")
    # last_conv_layer = model.get_layer("block5_pool")
    # encoder = Model(model.input,last_conv_layer.output)
    encoder = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=((224,224,3)))

    inputs1 = Input((224,224,3))
    inputs2 = Input((1,22))
    feature = encoder(inputs1)

    concate = tf.keras.Sequential()
    concate.add(Dense(7*7*512, activation="relu"))
    concate.add(Reshape((7, 7, 512)))
    au = concate(inputs2)
    feature_x_au = tf.concat((feature, au), axis=-1)

    x = Flatten()(feature_x_au)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="softmax")(x)

    clf = Model([inputs1,inputs2],x)
    clf.summary()
    return clf


class project_train():
    def __init__(self, batch, batch_num, epochs,validation_value):
        self.batch = batch
        self.batch_num = batch_num
        self.epochs = epochs
        # self.train_data, self.train_label, self.train_au, self.test_data, self.test_label, self.test_au = deal_data_overall()
        self.train_data, self.train_label, self.train_au = deal_data_overall()
        self.model = overall_model()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.validation_value = validation_value


    def train_step(self,inputs1,inputs2,label,train=True):
        with tf.GradientTape() as tape:
            pred = self.model([inputs1,inputs2])
            cce = tf.keras.losses.CategoricalCrossentropy()
            loss_clf = cce(label, pred)
            accuracy = accuracy_score(np.argmax(label, axis=1), np.argmax(pred, axis=1))
        if train:
            grads = tape.gradient(loss_clf,self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss_clf, accuracy


    def training(self):
        loss_clf_avg = []
        loss_te_avg = []
        accuracy_avg = []
        accuracy_te_avg = []

        # deal data
        count = 1
        train_data = []
        train_label = []
        train_au = []
        test_data = []
        test_label = []
        test_au = []
        for i in range(len(self.train_data)):
            if i != 0:
                if (i) % 210 == 0:
                    count += 1
            if (self.validation_value - 1 < count) and (count <= self.validation_value):
                test_data.append(self.train_data[i])
                test_label.append(self.train_label[i])
                test_au.append(self.train_au[i])
            else:
                train_data.append(self.train_data[i])
                train_label.append(self.train_label[i])
                train_au.append(self.train_au[i])
        train_data, train_label, train_au = np.array(train_data), np.array(train_label), np.array(train_au)
        test_data, test_label, test_au = np.array(test_data), np.array(test_label), np.array(test_au)
        print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)



        for epoch in range(self.epochs):
            start = time.time()
            loss_clf = []
            loss_te = []
            accuracy = []
            accuracy_te = []
            for b in range(self.batch_num):
                train, label, tr_au = get_batch_data(train_data, train_label, train_au, b, self.batch)
                test, te_label, te_au = get_batch_data(test_data, test_label, test_au, b , self.batch)
                tr_au, te_au = tr_au.reshape((-1,1,22)), te_au.reshape((-1,1,22))
                loss_c, acc = self.train_step(train, tr_au, label)
                try:
                    loss_t, acc_te = self.train_step(test,te_au,te_label)
                    accuracy_te.append(acc_te)
                    loss_te.append(loss_t)
                except:
                    pass
                loss_clf.append(loss_c)
                accuracy.append(acc)
            loss_clf_avg.append(np.mean(loss_clf))
            loss_te_avg.append(np.mean(loss_te))
            accuracy_avg.append(np.mean(accuracy))
            accuracy_te_avg.append(np.mean(accuracy_te))
            print("_______________________________")
            print(f"the epoch is {epoch+1}")
            print(f"the loss_clf is {loss_clf_avg[-1]}")
            print(f"the loss_test is {loss_te_avg[-1]}")
            print(f"the accuracy of train is {accuracy_avg[-1]}")
            print(f"the accuracy of test is {accuracy_te_avg[-1]}")
            print(f"the spend time is {time.time() - start}")
            if len(loss_te_avg) != 1:
                if loss_te_avg[-1] > loss_te_avg[-2]:
                    print(f"this model would be overfitting")
                    break
        self.model.save("overall_train.h5")
        return loss_clf_avg, accuracy_avg, accuracy_te_avg


def cross_validation():
    cross_acc = []
    for i in range(10):
        cnn =project_train(50,30,63,i+1)
        _, _, acc_val = cnn.training()
        cross_acc.append(acc_val[-1])
    st_dev = statistics.pstdev(cross_acc)
    print(cross_acc, st_dev)
    mean_acc = np.mean(cross_acc)
    print(f"mean +- std is {mean_acc}+-{st_dev}")



if __name__ == "__main__":
    cross_validation()
    # train = project_train(20,100,15)
    # loss_clf, accuracy, accuracy_te = train.training()
    #
    # plt.title("the loss_clf")
    # plt.plot(loss_clf)
    # plt.savefig("result_image/the loss_clf.jpg")
    # plt.close()
    #
    # plt.title("the accuracy")
    # plt.plot(accuracy)
    # plt.plot(accuracy_te)
    # plt.legend(["train","test"], loc="upper right")
    # plt.savefig("result_image/the accruacy.jpg")
    # plt.clo





