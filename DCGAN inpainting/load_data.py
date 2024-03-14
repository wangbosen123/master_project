import numpy as np
import cv2
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import collections


def load_path(train=True):
    data_path = []
    path = 'part of celebA'
    for filename in os.listdir(path):
        data_path.append(filename)
    train_path = data_path[0:8000]
    test_path = data_path[-100:]
    train_path = np.array(train_path)
    test_path = np.array(test_path)
    if train:
        return train_path
    else:
        return test_path


def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1 ) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min,range_max))
    train_data = [data[idx] for idx in index]
    return train_data

def load_image(roots,inpainting=False):
    train_data = []
    path = "part of celebA"
    for root in roots:
        img = cv2.imread(path + "/" + root,cv2.IMREAD_GRAYSCALE)
        if inpainting:
            img = cv2.rectangle(img,(22,22),(43,43),(0,0,0),-1)
        train_data.append(img)
    train_data = np.array(train_data)
    return (np.array(train_data).astype('float32')) / 127.5 - 1
'''
def make_mask():
    train_path = load_path(train=False)
    data = load_image(get_batch_data(train_path, 0, 1), inpainting=True)
    data = data.reshape(64, 64)
    mask = np.zeros((64,64))
    # 11*11
    for i in range(12, 64):
        for j in range(12, 64):
            window = data[i-11:i+11, j-11:j+11]
            window = window.reshape(window.shape[0] * window.shape[1])
            if data[i][j] == -1:
                mask[i][j] = 0
            else:
                mask[i][j] = collections.Counter(window)[-1] / 484
    return mask
'''

def make_mask():

    mask = np.zeros((64,64))
    # 11*11
    for i in range(32):
        for j in range(64):
          mask[i][j] += i/32
    return mask



if __name__ == "__main__":
    train_path = load_path(train=False)
    data = load_image(get_batch_data(train_path, 0, 1), inpainting=True)
    data = data.reshape(64, 64)
    # print(data[13][13])
    # import collections
    # sales = [0, 100, 100, 80, 70, 80, 20, 10, 100, 100, 80, 70, 10, 30, 40]
    # print(collections.Counter(data))
    # print(collections.Counter(data)[-1])
    # Counter({100: 4, 80: 3, 70: 2, 10: 2, 0: 1, 20: 1, 30: 1, 40: 1})
    # x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(x[0:1,1:2])
    # mask = make_mask()
    # print(mask[15][15])
    # data = load_image(get_batch_data(load_path(train=False), 0, 20))
    # data = (data+1)*127.5
    # cv2.imwrite("result.jpg", data[3])
    # plt.imshow(data[3],cmap="gray")
    # plt.savefig("result.jpg")

