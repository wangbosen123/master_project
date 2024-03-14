import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_path():
    train = []
    test = []
    train_path = "celeba_train"
    test_path = "celeba_test"
    for filename in os.listdir(train_path):
        train.append(filename)
    for filename in os.listdir(test_path):
        test.append(filename)
    train, test = np.array(train), np.array(test)
    return train, test



def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1 ) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min,range_max))
    train_data = [data[idx] for idx in index]
    return train_data


def load_image(roots,train=True):
    if train:
        path = "celeba_train"
    else:
        path = "celeba_test"

    low_resolution = []
    high_resolution = []
    for root in roots:
        image = cv2.imread(path + "/" + root, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(image,(5,5),sigmaX=1,sigmaY=1)
        low_img = cv2.resize(img, (8, 8), cv2.INTER_CUBIC)
        low_img = cv2.resize(low_img, (64, 64), cv2.INTER_CUBIC)
        low_resolution.append(low_img)
        high_resolution.append(image)
    low_resolution = np.array(low_resolution)/255
    high_resolution = np.array(high_resolution)/255
    return low_resolution, high_resolution


if __name__ == "__main__":
    path = load_path()
    print(path[0:20])

