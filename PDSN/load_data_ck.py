import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2

def load_path(train=True):
    train_path = []
    label = []
    if train:
        path_expression = "ck/train/Expression image"
        path_natural = "ck/train/Natural image"
    else:
        path_expression = "ck/train/Expression image"
        path_natural = "ck/test/Natural image"

    for id in os.listdir(path_expression):
        for filename in os.listdir(path_expression+"/"+id):
            for image_path in os.listdir(path_expression+"/"+id+"/"+filename):
                train_path.append(path_expression+"/"+id+"/"+filename+"/"+image_path)
                label.append(0)

    for id in os.listdir(path_natural):
        for filename in os.listdir(path_natural + "/" + id):
            for image_path in os.listdir(path_natural + "/" + id + "/" + filename):
                train_path.append(path_natural + "/" + id + "/" + filename + "/" + image_path)
                label.append(1)

    return np.array(train_path) , np.array(tf.one_hot(label,2))



def load_image(data_path, x1=80, x2=110, y1=40, y2=90):
    nocc_data, occ_data = [], []

    for i in range(len(data_path)):
        image = cv2.imread(data_path[i],cv2.IMREAD_GRAYSCALE)
        image = image/255
        image = np.expand_dims(image, axis=-1)
        nocc_data.append(image)

    for i in range(len(data_path)):
        image = cv2.imread(data_path[i], cv2.IMREAD_GRAYSCALE)
        image = image / 255
        image = np.expand_dims(image, axis=-1)
        image[x1:x2, y1:y2, :] = np.random.uniform(0, 1, size=(x2 - x1, y2 - y1, 1))
        occ_data.append(image)

    return np.array(nocc_data), np.array(occ_data)




def get_batch_data(nocc, occ, label, batch, batch_size):
    range_min = batch * batch_size
    range_max = (batch + 1) * batch_size
    if range_max > nocc.shape[0]:
        range_max = occ.shape[0]
    index = list(range(range_min, range_max))
    train_nocc = [nocc[idx] for idx in index]
    train_occ = [occ[idx] for idx in index]
    label = [label[idx] for idx in index]
    return  np.array(train_nocc), np.array(train_occ), np.array(label)




if __name__ == "__main__":
    train, label = load_path(train=True)
    train_nocc, train_occ = load_image(train,80,110,40,90)
    print(train_nocc.shape, train_occ.shape)
    nocc, occ , label = get_batch_data(train_nocc, train_occ, label, 41, 20)
    print(nocc.shape,occ.shape,label.shape)



    # ax,fig = plt.subplots(figsize=(15,4))
    # for i in range(10):
    #     plt.subplot(2,10,i+1)
    #     plt.axis("off")
    #     plt.imshow(train_nocc[i], cmap="gray")
    #     plt.subplot(2,10,i+11)
    #     plt.axis("off")
    #     plt.imshow(train_occ[i], cmap="gray")
    # plt.savefig("image/train_data.jpg")






