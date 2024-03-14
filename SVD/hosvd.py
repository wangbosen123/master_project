import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import math
import tensorly as tl
from tensorly.decomposition import tucker
import pandas  as pd



def get_tensor(digit):
    count = 0
    step = 0
    tensor = np.zeros((28,28,10000))
    for filename in os.listdir("train"):
        if int(filename[0]) != digit:
            continue
        count += 1
        image = cv2.imread("train/" + filename, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        tensor[:,:,step] = image
        step+=1
        if count == 1000:
            break
    return tensor


def get_s_u():
    s_list = []
    u_list = []
    for i in range(10):
        print(i)
        tensor = get_tensor(i)
        core, factors = tucker(tensor, rank=[28,28,1000])
        s_list.append(core)
        u_list.append(factors)
    s_list, u_list = np.array(s_list), np.array(u_list)
    return s_list, u_list

def single_test(image, k, s_list, u_list):
    residual = []
    compare_image = []
    for i in range(10):
        aj_final = np.zeros((28, 28))
        for j in range(k+1):
            aj = u_list[i][0].dot(s_list[i][:,:,j]).dot(u_list[i][1].T)
            zj = np.tensordot(image, aj) / np.tensordot(aj, aj)
            aj_final += zj * aj
        compare_image.append(aj_final)
        residual.append(np.linalg.norm(image - aj_final, "fro"))
    pred = np.argmin(residual)
    return pred, compare_image, residual


def test_acc(s_list, u_list, k):
    wrong = []
    wrong_pred = []
    label = []
    acc = 0
    # total = []
    for filename in os.listdir("test"):
        image = cv2.imread("test"+"/"+filename, cv2.IMREAD_GRAYSCALE)
        # if int(filename[0]) != 9:
        #     continue
        image = image / 255
        residual = []
        for i in range(10):
            aj_final = np.zeros((28, 28))
            for j in range(k + 1):
                aj = u_list[i][0].dot(s_list[i][:, :, j]).dot(u_list[i][1].T)
                zj = np.tensordot(image, aj) / np.tensordot(aj, aj)
                aj_final += zj * aj
            residual.append(np.linalg.norm(image - aj_final, "fro"))
        # total.append(residual)
        pred = np.argmin(residual)
        if pred == int(filename[0]):
            acc += 1
        else:
            wrong.append(image)
            wrong_pred.append(pred)
            label.append(int(filename[0]))
    acc /= 2000
    return acc, wrong, wrong_pred, label#, total




if __name__ == "__main__":
    image = cv2.imread("test3.png",cv2.IMREAD_GRAYSCALE)
    image = image / 255
    img = image.reshape(784)
    # s_list, u_list = get_s_u()

    ##single
    # pred, compare, residual = single_test(image, 8, s_list, u_list)
    # print(pred, residual)
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



    #all test
    # acc, wrong, wrong_pred, label = test_acc(s_list, u_list, 18)
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


    # different k for acc
    # acc_total = []
    # for i in range(30):
    #     print(i)
    #     acc, wrong, wrong_pred, label = test_acc(s_list,u_list, i)
    #     acc_total.append(acc)
    #
    # plt.plot(acc_total)
    # plt.xlabel("k")
    # plt.ylabel("accuracy")
    # plt.savefig("result.jpg")
    # plt.close()



    #number of correct digit for testing
    # number = [[] for i in range(10)]
    # acc, wrong, wrong_pred, label = test_acc(s_list,u_list, 18)
    # print(acc)
    # for i in wrong_pred:
    #     number[i].append(0)
    # for i in range(10):
    #     print(len(number[i]))



    # draw
    # acc, wrong, wrong_pred, label, total = test_acc(s_list,u_list, 18)
    # for i in range(200):
    #     plt.plot(total[i])
    # plt.xlim(0,9)
    # plt.ylabel("residual")
    # plt.savefig("result.jpg")
    # plt.close()

    # draw the S
    tensor = get_tensor(7)
    core, factors = tucker(tensor, rank=[28, 28, 1000])
    plt.subplots(figsize=(15, 4))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis("off")
        plt.title(f"S[:,:,{i}]")
        plt.imshow(core[:,:,i],cmap="gray")
    plt.savefig("result.jpg")
    plt.close()








    # tensor = get_tensor(2)
    # core, factors = tucker(tensor, rank=[28, 28, 1000])
    # aj_final = np.zeros((28, 28))
    # for j in range(12):
    #     aj = factors[0].dot(core[:, :, j]).dot(factors[1].T)
    #     zj = np.tensordot(image, aj) / np.tensordot(aj, aj)
    #     aj_final += zj * aj
    # plt.imshow(aj_final, cmap="gray")
    # plt.show()
