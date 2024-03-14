import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from build_model import *
import tensorflow_datasets as tfds


def load_path():
    smile_path = []
    nature_path = []
    domaina_path = "ck_nature"
    domainb_path = "ck_smile"
    for filename in os.listdir(domainb_path):
        smile_path.append(domainb_path+"/"+filename)
    for filename in os.listdir(domaina_path):
        nature_path.append(domaina_path+"/"+filename)
    nature_test_path = nature_path[660:]
    smile_path, nature_path, neture_test_path = np.array(smile_path[0:660]), np.array(nature_path[0:660]), np.array(nature_test_path)
    return nature_path, smile_path, nature_test_path




def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1 ) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min,range_max))
    train_data = [data[idx] for idx in index]
    return train_data


def load_image(roots):
    data = []
    for root in roots:
        image =cv2.imread(root, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        data.append(image)
    data = np.array(data)
    return data



if __name__ == "__main__":
    pass
    # path_smile, path_nature = load_path()
    #
    # domain_a = load_image(get_batch_data(path_nature, 0, 2))
    # domain_b = load_image(get_batch_data(path_smile, 0, 2))
    # domain_a, domain_b = domain_a.reshape(-1,128,128,1), domain_b.reshape(-1,128,128,1)
    # print(domain_a.shape, domain_b.shape)
    # data = np.concatenate((domain_a, domain_b), axis=-1)
    # print(data[:,:,:,0].shape)
