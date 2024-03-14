import numpy as np
import os
import cv2
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

def deal_data_overall(path="0412_emotion_dataset"):
    image_set = []
    label = []
    au_encoding = []
    emos = ["0_frustration", "4_flow", "5_suprise", "2_bored", "3_happy", "1_confused"]

    for emo in emos:
        for filename in os.listdir(path+"/"+emo):
            image = cv2.imread(path + "/" + emo + "/" + filename)/255
            image_set.append(image)
            label.append(int(emo[0]))

    for filename in os.listdir((path+"/" + "au_encoding")):
        file = open(f"0412_emotion_dataset/au_encoding/{filename}")
        reader = csv.reader(file)
        data_list = list(reader)
        for i in range(1, len(data_list)):
            au = []
            for j in range(len(data_list[i][3])):
                try:
                    au.append(int(data_list[i][3][j]))
                except:
                    continue
            au_encoding.append(au)
        file.close()

    data = list(zip(image_set, label, au_encoding))
    np.random.shuffle(data)
    data = list(zip(*data))

    return np.array(data[0][0:2100]), tf.one_hot(np.array(data[1][0:2100]),6), np.array(data[2][0:2100])

# train, test, au = deal_data_overall()
# print(train.shape, test.shape, au.shape)

def deal_data(path="0412_emotion_dataset"):
    image_set = []
    label = []

    for emo in os.listdir(path):
        if emo == "au_encoding":
            continue

        for filename in os.listdir(path+"/"+emo):
            image = cv2.imread(path + "/" + emo + "/" + filename)/255
            image_set.append(image)
            label.append(int(emo[0]))

    data = list(zip(image_set, label))
    np.random.shuffle(data)
    data = list(zip(*data))

    return np.array(data[0][0:2100]), tf.one_hot(np.array(data[1][0:2100]),6)



def get_batch_data(data,label,au,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max  = (batch_idx + 1) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min,range_max))
    temp_data = [data[idx] for idx in index]
    temp_label = [label[idx] for idx in index]
    temp_au = [au[idx] for idx in index]
    return np.array(temp_data), np.array(temp_label), np.array(temp_au)

def get_batch_data_cnn(data,label,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max  = (batch_idx + 1) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min,range_max))
    temp_data = [data[idx] for idx in index]
    temp_label = [label[idx] for idx in index]
    return np.array(temp_data), np.array(temp_label)

#use the fer_dataset

def deal_csv():
    file = open(f"fer2013.xls")
    reader = csv.reader(file)
    data_list = list(reader)


    data = []
    filename = 0
    for i in range(1,len(data_list)):
        print(i)
        image_value = []
        filename += 1
        for value in data_list[i][1]:
            image_value.append(value)

        count = 0
        length = 0
        image = []
        for pixel in image_value:
            try:
                int(pixel)
                count = count * 10 + int(pixel)
                if length == len(image_value)-1:
                    image.append(count)
            except:
                image.append(count)
                count = 0
            length += 1
        image = np.array(image)
        image = image.reshape((48,48))
        cv2.imwrite(f"fer_dataset/{int(data_list[i][0])}/{filename}.jpg",image)


def spilt_train_test(path="fer_dataset"):
    data = []
    label = []
    for folder in os.listdir("fer_dataset"):
        for filename in os.listdir("fer_dataset"+"/"+folder):
            data.append(cv2.imread("fer_dataset"+"/"+folder+"/"+filename))
            label.append(int(folder))

    image = list(zip(data,label))
    np.random.shuffle(image)
    image = list(zip(*image))

    train, train_label = image[0][0:31000], image[1][0:31000]
    test, test_label = image[0][31000:], image[1][31000:]
    train, train_label, test, test_label = np.array(train), np.array(train_label), np.array(test), np.array(test_label)

    count = 0
    for i in range(len(train)):
        count += 1
        cv2.imwrite(f"train1/{train_label[i]}/{count}.jpg",train[i])
        print(count)


    count = 0
    for i in range(len(test)):
        count+=1
        cv2.imwrite(f"test1/{test_label[i]}/{count}.jpg", test[i])
        print(count)


def deal_data_fer(path="train1"):
    image_set = []
    label = []

    for emo in os.listdir(path):
        for filename in os.listdir(path+"/"+emo):
            image = cv2.imread(path+"/"+emo+"/"+filename)
            image = image/255
            image_set.append(image)
            label.append(emo)

    data = list(zip(image_set, label))
    np.random.shuffle(data)
    data = list(zip(*data))

    return np.array(data[0][0:28000]), tf.one_hot(np.array(data[1][0:28000]),7), np.array(data[0][28000:]), tf.one_hot(np.array(data[1][28000:]),7)

