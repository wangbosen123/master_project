import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

def get_mean():
    mean = [np.zeros((28,28)) for i in range(10)]
    for filename in os.listdir("train"):
        image = cv2.imread("train" + "/" + filename, cv2.IMREAD_GRAYSCALE)
        image = image/255
        mean[int(filename[0])] += image

    mean = np.array(mean) / 1000
    return mean


def prediction(target,mean):
    loss = []
    for i in range(len(mean)):
        loss.append(np.sqrt(np.sum(np.square(target - mean[i]))))

    loss = np.array(loss)
    return np.where([loss == min(loss)])[1][0], loss


def test_acc():
    wrong = []
    wrong_pred = []
    label = []
    acc = 0
    mean = get_mean()
    for filename in os.listdir("test"):
        image = cv2.imread("test/"+filename,cv2.IMREAD_GRAYSCALE) / 255
        pred, _ = prediction(image, mean)
        if pred == int(filename[0]):
            acc += 1
            # print(acc)
        else:
            wrong.append(image)
            label.append(int(filename[0]))
            wrong_pred.append(pred)
    acc /= 2000
    return acc, wrong, wrong_pred, label



# mean method  1.get the mean of matrix 1.1 number of pixel value  2.test the number use the mean method  3.use self handwrite  4. wrong answer
if __name__ == "__main__":
    image = cv2.imread("test3.png", cv2.IMREAD_GRAYSCALE)
    image = image /255
    mean = get_mean()
    pred, loss = prediction(image,mean)
    print(pred, loss)

    acc, wrong, wrong_pred, label = test_acc()
    test = [[] for i in range(10)]
    for i in range(len(wrong)):
        test[wrong_pred[i]].append(0)
    print(len(test))

    # print(acc)
    # wrong = np.array(wrong)
    # plt.subplots(figsize=(15, 4))
    # for i in range(10):
    #     plt.subplot(2, 5, i + 1)
    #     plt.axis("off")
    #     plt.title(f"label {label[i + 30]}, pred {wrong_pred[i + 30]}")
    #     plt.imshow(wrong[i + 30], cmap="gray")
    #     # plt.imshow(get_image_and_loss(image,i),cmap="gray")
    # plt.savefig("result.jpg")



    # mean = get_mean()
    #
    # plt.subplots(figsize=(15,4))
    # count = 0
    # for i in range(10):
    #     plt.subplot(2,5,i+1)
    #     plt.axis("off")
    #     plt.title(f"label {count}")
    #     plt.imshow(mean[i], cmap="gray")
    #     count += 1
    # plt.savefig("result.jpg")
    # plt.close()























