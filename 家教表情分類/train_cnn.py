import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import os
import matplotlib.pyplot as plt
import cv2
from load_data import *
import time
from sklearn.metrics import accuracy_score
import statistics


class train_cnn():
    def __init__(self,batch_size, batch_num, epochs,validation_value):
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.epochs = epochs
        self.model = model()
        # self.train_data, self.train_label,self.val_data, self.val_label = deal_data()
        self.train_data, self.train_label = deal_data()
        self.optimizer = tf.keras.optimizers.Adam(1e-5)
        self.validation_value = validation_value

    def train_step(self, inputs, label, train=True):
        with tf.GradientTape() as tape:
            pred = self.model(inputs)
            cce = tf.keras.losses.CategoricalCrossentropy()
            loss_clf = cce(label, pred)
            acc = accuracy_score(np.argmax(label,axis=1),np.argmax(pred,axis=1))
        if train:
            grads = tape.gradient(loss_clf, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_clf, acc

    def training(self):
        loss_clf_avg = []
        loss_te_avg = []
        acc_avg = []
        acc_te_avg = []

        #deal data
        count = 1
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(self.train_data)):
            if i != 0:
                if (i) % 210 == 0:
                    count +=1
            if (self.validation_value-1 <count) and (count<= self.validation_value):
                test_data.append(self.train_data[i])
                test_label.append(self.train_label[i])
            else:
                train_data.append(self.train_data[i])
                train_label.append(self.train_label[i])
        train_data, train_label = np.array(train_data), np.array(train_label)
        test_data, test_label = np.array(test_data), np.array(test_label)
        print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)


        for epoch in range(self.epochs):
            start = time.time()
            loss_clf = []
            loss_val = []
            acc = []
            acc_te = []
            for batch in range(self.batch_num):
                train, label = get_batch_data_cnn(train_data, train_label, batch, self.batch_size)
                test, te_label = get_batch_data_cnn(test_data, test_label, batch, self.batch_size)

                loss_c, accuracy = self.train_step(train,label,train=True)
                try:
                    loss_v, accuracy_te = self.train_step(test,te_label, train=False)
                    acc_te.append(accuracy_te)
                    loss_val.append(loss_v)
                except:
                    pass
                loss_clf.append(loss_c)
                acc.append(accuracy)
            loss_clf_avg.append(np.mean(loss_clf))
            loss_te_avg.append(np.mean(loss_val))
            acc_avg.append(np.mean(acc))
            acc_te_avg.append(np.mean(acc_te))
            print("________________________________________")
            print(f"the spend time is {time.time() - start}")
            print(f"the epoch is {epoch+1}")
            print(f"the loss_clf is {loss_clf_avg[-1]}")
            print(f"the loss_val is {loss_te_avg[-1]}")
            print(f"the accuracy of train is {acc_avg[-1]}")
            print(f"the accuracy of validation is {acc_te_avg[-1]}")
            if len(loss_te_avg) != 1:
                if loss_te_avg[-1] > loss_te_avg[-2]:
                    print(f"this model would be overfitting")
                    break
        self.model.save("pretrain_fer.h5")
        return loss_clf_avg, loss_te_avg, acc_avg, acc_te_avg



def model():
    model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=((224,224,3)))
    x = Flatten()(model.output)
    x = Dense(256,activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="softmax")(x)

    model = Model(model.input, x)
    model.summary()
    return model

def cross_validation():
    cross_acc = []
    for i in range(10):
        cnn = train_cnn(50,30,63,i+1)
        _, _, _, acc_val = cnn.training()
        cross_acc.append(acc_val[-1])
    st_dev = statistics.pstdev(cross_acc)
    print(cross_acc, st_dev)
    mean_acc = np.mean(cross_acc)
    print(f"mean +- std is {mean_acc}+-{st_dev}")

if __name__ == "__main__":
    cross_validation()




    # cnn = train_cnn(50, 30, 63, 2)
    # loss_clf, loss_val, accuracy, acc_val = cnn.training()
    #
    # plt.title("the loss")
    # plt.plot(loss_clf)
    # plt.plot(loss_val)
    # plt.legend(["train", "validation"], loc="upper right")
    # plt.savefig("result_image/the loss.jpg")
    # plt.close()
    #
    # plt.title("the accuracy")
    # plt.plot(accuracy)
    # plt.plot(acc_val)
    # plt.legend(["train","validation"], loc="upper right")
    # plt.savefig("result_image/the accruacy.jpg")
    # plt.close()











