import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import math

def get_uut(digit, k):
    count = 0
    data = []
    for filename in os.listdir("train"):
        if int(filename[0]) != digit:
            continue
        count += 1
        image = cv2.imread("train/" + filename, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        image = image.reshape(784)
        data.append(image)
        if count == 1000:
            break

    data = np.array(data)
    data = data.T

    u, s, v = np.linalg.svd(data, full_matrices=1)
    uut = u[:, :k+1].dot(u[:, :k+1].T)
    return uut


def get_image_and_loss(target, digit):
    count = 0
    data = []
    for filename in os.listdir("train"):
        if int(filename[0]) != digit:
            continue
        count += 1
        image = cv2.imread("train/" + filename, cv2.IMREAD_GRAYSCALE)
        image = image /255
        image = image.reshape(784)
        data.append(image)
        if count == 1000:
            break

    data = np.array(data)
    data = data.T

    u, s, v = np.linalg.svd(data, full_matrices=1)
    uutz = np.dot(u[:, :12].dot(u[:, :12].T), target)
    residual = target - uutz
    residual = np.sqrt(np.sum(np.square(residual)))
    uutz = uutz.reshape(28,28)
    return uutz, residual

def single_pred_and_loss(target):
    total_loss = []
    for i in range(10):
        pred, loss = get_image_and_loss(target,i)
        total_loss.append(loss)
    total_loss = np.array(total_loss)

    return np.where([total_loss == min(total_loss)])[1][0], total_loss

def test_acc(k):
    wrong = []
    wrong_pred = []
    all_uut = []
    label = []
    acc = 0
    # total = []
    for i in range(10):
        all_uut.append(get_uut(i,k))

    for filename in os.listdir("test"):
        # if int(filename[0]) != 6:
        #     continue
        img = cv2.imread("test/" + filename, cv2.IMREAD_GRAYSCALE) / 255
        image = img.reshape(784)
        total_loss = []
        for i in range(10):
            total_loss.append(np.sum(np.square(image - np.dot(all_uut[i], image))))
        # total.append(total_loss)
        pred = np.where([total_loss == min(total_loss)])[1][0]
        if pred == int(filename[0]):
            acc += 1
        else:
            wrong.append(img)
            wrong_pred.append(pred)
            label.append(int(filename[0]))
    acc /= 2000
    return acc, wrong, wrong_pred, label#, total


if __name__ == "__main__":
    image = cv2.imread("test4.png", cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = image.reshape(784)

    #draw the compare
    # compare = []
    # for i in range(10):
    #     pic,_ = get_image_and_loss(image,i)
    #     compare.append(pic)
    # count = 0
    # plt.subplots(figsize=(15,4))
    # for i in range(10):
    #     plt.subplot(2,5,i+1)
    #     plt.axis("off")
    #     plt.title(f"label {count}")
    #     plt.imshow(compare[i], cmap="gray")
    #     count+=1
    #     # plt.imshow(get_image_and_loss(image,i),cmap="gray")
    # plt.savefig("result.jpg")

    #single test
    # pred, loss = single_pred_and_loss(image)
    # print(pred, loss)



    #all test
    # acc, wrong, wrong_pred, label = test_acc()
    # print(acc)
    # print(len(wrong_pred))
    # wrong = np.array(wrong)
    # plt.subplots(figsize=(15,4))
    # for i in range(10):
    #     plt.subplot(2,5,i+1)
    #     plt.axis("off")
    #     plt.title(f"label {label[i+30]}, pred {wrong_pred[i+30]}")
    #     plt.imshow(wrong[i+30], cmap="gray")
    #     # plt.imshow(get_image_and_loss(image,i),cmap="gray")
    # plt.savefig("result.jpg")

    #different k for acc
    # acc_total = []
    # for i in range(30):
    #     print(i)
    #     acc, wrong, wrong_pred, label = test_acc(i)
    #     acc_total.append(acc)
    #
    # plt.plot(acc_total)
    # plt.xlabel("k")
    # plt.ylabel("accuracy")
    # plt.savefig("result.jpg")
    # plt.close()
    #
    #number of correct digit for testing
    # number = [[] for i in range(10)]
    # acc, wrong, wrong_pred, label = test_acc(18)
    # for i in wrong_pred:
    #     number[i].append(0)
    # for i in range(10):
    #     print(len(number[i]))

    #draw
    # acc, wrong, wrong_pred, label, total = test_acc(18)
    # for i in range(200):
    #     plt.plot(total[i])
    # plt.xlim(0,9)
    # plt.ylabel("residual")
    # plt.savefig("result.jpg")
    # plt.close()




