import os
import numpy as np
import tensorflow as tf
import cv2
from tool import *
import matplotlib.pyplot as plt

def restore_data():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    path = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_test21/"
    data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.02)])
    for ID in os.listdir(path):
        res = 0
        print(ID)
        for filename in os.listdir(path + ID):
            images_gt, images_h, images_m, images_l = [], [], [], []
            if '-14-' in filename:
                gray_gt = cv2.imread(path + ID + '/' + filename, 0)
                gray_h = cv2.resize(cv2.resize(gray_gt, (int(gray_gt.shape[1]/2), int(gray_gt.shape[0]/2)), cv2.INTER_CUBIC), (gray_gt.shape[1], gray_gt.shape[0]), cv2.INTER_CUBIC)
                gray_m = cv2.resize(cv2.resize(gray_gt, (int(gray_gt.shape[1]/4), int(gray_gt.shape[0]/4)), cv2.INTER_CUBIC), (gray_gt.shape[1], gray_gt.shape[0]), cv2.INTER_CUBIC)
                gray_l = cv2.resize(cv2.resize(gray_gt, (int(gray_gt.shape[1]/8), int(gray_gt.shape[0]/8)), cv2.INTER_CUBIC), (gray_gt.shape[1], gray_gt.shape[0]), cv2.INTER_CUBIC)

                images_gt.append(gray_gt)
                images_h.append(gray_h)
                images_m.append(gray_m)
                images_l.append(gray_l)

                for i in range(5):
                    image_gt_aug = data_augmentation(gray_gt.reshape(1, gray_gt.shape[0], gray_gt.shape[1], 1))
                    image_h_aug = data_augmentation(gray_h.reshape(1, gray_h.shape[0], gray_h.shape[1], 1))
                    image_m_aug = data_augmentation(gray_m.reshape(1, gray_m.shape[0], gray_m.shape[1], 1))
                    image_l_aug = data_augmentation(gray_l.reshape(1, gray_l.shape[0], gray_l.shape[1], 1))
                    images_gt.append(tf.reshape(image_gt_aug, [gray_gt.shape[0], gray_gt.shape[1]]).numpy())
                    images_h.append(tf.reshape(image_h_aug, [gray_h.shape[0], gray_h.shape[1]]).numpy())
                    images_m.append(tf.reshape(image_m_aug, [gray_m.shape[0], gray_m.shape[1]]).numpy())
                    images_l.append(tf.reshape(image_l_aug, [gray_l.shape[0], gray_l.shape[1]]).numpy())
                images_1, images_2, images_3 = np.array(images_h), np.array(images_m), np.array(images_l)
                print(images_1.shape, images_2.shape, images_3.shape)
                images = [images_h, images_m, images_l]
                for reso, reso_images in enumerate(images):
                    for num, image in enumerate(reso_images):
                        faces = face_cascade.detectMultiScale(image, 1.1, 2, minSize=(70, 70))
                        if len(faces) == 0:
                            break
                        for faces_num, (x, y, w, h) in enumerate(faces):
                            if faces_num == 1:
                                image = image[y:y + h, x:x + w]
                                break
                            image = image[y:y + h, x:x + w]

                        try:
                            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
                            if reso == 0:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system/alignment_test_data/{ID}/{res + 1}_2_ratio.jpg', image)
                                res += 1
                            if reso == 1:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system/alignment_test_data/{ID}/{res + 1}_4_ratio.jpg', image)
                                res += 1
                            if reso == 2:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system/alignment_test_data/{ID}/{res + 1}_8_ratio.jpg', image)
                                res += 1
                        except:
                            continue

def restore_data_original():
    path = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_train90/"
    data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.02)])
    for ID in os.listdir(path):
        res = 0
        print(ID)
        for filename in os.listdir(path + ID):
            images_gt, images_h, images_m, images_l = [], [], [], []
            if '-14-' in filename:
                gray_gt = cv2.imread(path + ID + '/' + filename, 0)
                for i in range(5):
                    image_gt_aug = data_augmentation(gray_gt.reshape(1, gray_gt.shape[0], gray_gt.shape[1], 1))
                    image_h_aug = tf.image.resize(tf.image.resize(image_gt_aug, [int(gray_gt.shape[1]/2), int(gray_gt.shape[0]/2)], method='bicubic'), [gray_gt.shape[0], gray_gt.shape[1]], method='bicubic')
                    image_m_aug = tf.image.resize(tf.image.resize(image_gt_aug, [int(gray_gt.shape[1]/4), int(gray_gt.shape[0]/4)], method='bicubic'), [gray_gt.shape[0], gray_gt.shape[1]], method='bicubic')
                    image_l_aug = tf.image.resize(tf.image.resize(image_gt_aug, [int(gray_gt.shape[1]/8), int(gray_gt.shape[0]/8)], method='bicubic'), [gray_gt.shape[0], gray_gt.shape[1]], method='bicubic')
                    images_gt.append(tf.reshape(image_gt_aug, [image_gt_aug.shape[1], image_gt_aug.shape[2]]).numpy())
                    images_h.append(tf.reshape(image_h_aug, [image_gt_aug.shape[1], image_gt_aug.shape[2]]).numpy())
                    images_m.append(tf.reshape(image_m_aug, [image_gt_aug.shape[1], image_gt_aug.shape[2]]).numpy())
                    images_l.append(tf.reshape(image_l_aug, [image_gt_aug.shape[1], image_gt_aug.shape[2]]).numpy())
                images_gt, images_h, images_m, images_l = np.array(images_gt), np.array(images_h), np.array(images_m), np.array(images_l)
                print(images_gt.shape, images_h.shape, images_m.shape, images_l.shape)
                images = [images_gt, images_h, images_m, images_l]
                for reso, reso_images in enumerate(images):
                    print(reso)
                    for num, image in enumerate(reso_images):
                        try:
                            print(res + num + 1)
                            if reso == 0:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system_0402/alignment_train_data/{ID}/{(res*6) + num + 1}_gt.jpg', image)
                            if reso == 1:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system_0402/alignment_train_data/{ID}/{(res*6) + num + 1}_2_ratio.jpg', image)
                            if reso == 2:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system_0402/alignment_train_data/{ID}/{(res*6) + num + 1}_4_ratio.jpg', image)
                            if reso == 3:
                                cv2.imwrite(f'/home/bosen/gradation_thesis/correlation_system_0402/alignment_train_data/{ID}/{(res*6) + num + 1}_8_ratio.jpg', image)

                        except:
                            continue
                res += 1

def load_data_path(shuffle=False):
    reference_path = '/home/bosen/PycharmProjects/Datasets/AR_train/'
    train_path = 'alignment_train_data/'
    train, reference, label = [], [], []
    train_number = [0 for i in range(90)]
    ID = [i for i in range(1, 91)]

    for num, id in enumerate(ID):
        for filename in os.listdir(train_path + f'ID{id}'):
            if 'gt' not in filename:
                train.append(train_path + f'ID{id}' + '/' + filename)
                train_number[num] += 1
                label.append(tf.one_hot(id - 1, 90))

    for num, id in enumerate(ID):
        for count, filename in enumerate(os.listdir(reference_path + f'ID{id}')):
            if count == 2:
                for i in range(train_number[num]):
                    reference.append(reference_path + f'ID{id}' + '/' + filename)
    if shuffle:
        data = list(zip(train, reference, label))
        np.random.shuffle(data)
        data = list(zip(*data))
        return np.array(data[0]), np.array(data[1]), np.array(data[2])
    else:
        return np.array(train), np.array(reference), np.array(label)

def load_test_data_path(shuffle=False):
    reference_path = "/home/bosen/gradation_thesis/correlation_system/ar_test/"
    test_path = 'alignment_test_data/'
    test, reference, label = [], [], []
    test_number = [0 for i in range(21)]
    ID = [i for i in range(1, 22)]

    for num, id in enumerate(ID):
        for filename in os.listdir(test_path + f'ID{id}'):
            if 'gt' not in filename:
                test.append(test_path + f'ID{id}' + '/' + filename)
                test_number[num] += 1
                label.append(tf.one_hot(id - 1, 90))

    for num, id in enumerate(ID):
        for count, filename in enumerate(os.listdir(test_path + f'ID{id}')):
            if count == 0:
                for i in range(test_number[num]):
                    reference.append(test_path + f'ID{id}' + '/' + filename)
    if shuffle:
        data = list(zip(test, reference, label))
        np.random.shuffle(data)
        data = list(zip(*data))
        return np.array(data[0]), np.array(data[1]), np.array(data[2])
    else:
        return np.array(test), np.array(reference), np.array(label)

def get_batch_data(train_data, gt_data, labels, batch_idx, batch_size):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1) * batch_size

    if range_max > len(gt_data):
        range_max = len(gt_data)
    index = list(range(range_min, range_max))
    gt_data = [gt_data[idx] for idx in index]
    train_data = [train_data[idx] for idx in index]
    labels = [labels[idx] for idx in index]

    gt_images, train_images, points, train_label = [], [], [], []
    images = list(zip(gt_data, train_data, labels))
    for num, (gt_path, train_path, label) in enumerate(images):
        train_gray = cv2.imread(train_path, 0)
        faces = face_cascade.detectMultiScale(train_gray, 1.1, 2, minSize=(70, 70))
        if len(faces) == 0:
            continue
        points.append([])
        for faces_num, (x, y, w, h) in enumerate(faces):
            if faces_num == 1:
                points[-1].append(x)
                points[-1].append(y)
                points[-1].append(w)
                points[-1].append(h)
                break
            points[-1].append(x)
            points[-1].append(y)
            points[-1].append(w)
            points[-1].append(h)

        gt_gray = cv2.imread(gt_path, 0) / 255
        if '2_ratio' in train_path:
            gt_gray = cv2.resize(gt_gray, (32, 32), cv2.INTER_CUBIC)
        if '4_ratio' in train_path:
            gt_gray = cv2.resize(gt_gray, (16, 16), cv2.INTER_CUBIC)
        if '8_ratio' in train_path:
            gt_gray = cv2.resize(gt_gray, (8, 8), cv2.INTER_CUBIC)
        gt_gray = cv2.resize(gt_gray, (64, 64), cv2.INTER_CUBIC)

        train_images.append(train_gray.reshape(train_gray.shape[0], train_gray.shape[1], 1)/255)
        gt_images.append(gt_gray.reshape(64, 64, 1))
        train_label.append(label)
    return np.array(train_images), np.array(gt_images), np.array(points), np.array(train_label)


if __name__ == "__main__":
    pass



