import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import dlib
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *

def aligment():
    # file_path = "/home/bosen/PycharmProjects/Datasets/AR_train/ID03/"
    file_path = f'AR_original_data/dbf1/'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/bosen/PycharmProjects/Datasets/shape_predictor_68_face_landmarks.dat")
    count = 0
    for num, filename in enumerate(os.listdir(file_path)):
        if 'bmp' not in filename:
            continue
        img = cv2.imread(file_path + '/' + filename, cv2.IMREAD_GRAYSCALE)
        x1 = 61
        y1 = 76
        x2 = 43
        y2 = 40
        x3 = 81
        y3 = 41
        rects = detector(img, 0)
        z1 = []
        z2 = []
        if len(rects) != 0:
            count += 1
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
            xx1 = landmarks[33][0, 0]
            yy1 = landmarks[33][0, 1]
            xx2 = landmarks[39][0, 0]
            yy2 = landmarks[39][0, 1]
            xx3 = landmarks[42][0, 0]
            yy3 = landmarks[42][0, 1]
            print(xx1, yy1, xx2, yy2, xx3, yy3)

            pts1 = np.float32([[xx1, yy1], [xx2, yy2], [xx3, yy3]])
            pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
            M = cv2.getAffineTransform(pts1, pts2)
            res = cv2.warpAffine(img, M, (128, 128))
            # plt.scatter(pts2[:, 0:1], pts2[:, 1:2], color='r')
            # plt.imshow(res, cmap='gray')
            # plt.axis('off')
            # plt.show()
            # res = cv2.equalizeHist(res)
            # plt.imshow(res, cmap='gray')
            # plt.axis('off')
            # plt.show()
            # cv2.imwrite(f"AR_aligment_initial/dbf1/{filename}", res)

        #     for idx, point in enumerate(landmarks):
        #         z1.append(point[0, 0])
        #         z2.append(point[0, 1])
        # re = cv2.re size(img[min(z2)-5:max(z2)-5, min(z1):max(z1)], (64, 64), interpolation=cv2.INTER_CUBIC)
        print(count)

def classifier(train=False):
    input = Input((128, 128, 1))
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(input)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = Conv2D(64, 3, padding='same', strides=1, activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Flatten()(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.4)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(0.4)(out)
    out = Dense(90, activation='softmax')(out)
    model = Model(input, out)
    model.summary()
    if train:
        file_path = "/home/bosen/PycharmProjects/Datasets/AR_train/"
        images, labels = [], []
        for ID in os.listdir(file_path):
            for filename in os.listdir(file_path + ID):
                image = cv2.imread(file_path + ID + '/' + filename, 0)
                labels.append(tf.one_hot(int(ID[2:]) - 1, 90))
                images.append(image / 255)
        images, labels = np.array(images), np.array(labels)
        data = list(zip(images, labels))
        np.random.shuffle(data)
        data = list(zip(*data))
        model.compile(optimizer=Adam(1e-4), loss='CategoricalCrossentropy', metrics=['accuracy'])
        model.fit(np.array(data[0]), np.array(data[1]), batch_size=30, epochs=10, verbose=1, validation_split=0.1)
        model.save('cls.h5')
    else:
        return model

def cls_aligment_data():
    cls = load_model('cls.h5')
    path = 'AR_aligment_initial/dbf5/'
    for i in range(1,76):
        try:
            if i > 9:
                image = cv2.imread(path + f'm-0{i}-14-0.bmp', 0)
            else:
                image = cv2.imread(path + f'm-00{i}-14-0.bmp', 0)
            pred = cls(image.reshape(1, 128, 128, 1)/255)
            print(f'{i}_th person is {np.argmax(pred, axis=-1)}, Probality is {pred[0][np.argmax(pred, axis=-1)[0]]}')
        except:
            pass
    print('----------------------------------------------------------')
    for i in range(1, 60):
        try:
            if i > 9:
                image = cv2.imread(path + f'w-0{i}-14-0.bmp', 0)
            else:
                image = cv2.imread(path + f'w-00{i}-14-0.bmp', 0)
            pred = cls(image.reshape(1, 128, 128, 1) / 255)
            print(f'{i}_th person is {np.argmax(pred, axis=-1)}, Probality is {pred[0][np.argmax(pred, axis=-1)[0]]}')
        except:
            pass

# aligment()

# cls_aligment_data()

# path = 'AR_aligment_initial/dbf1/'
# for num, file in enumerate(os.listdir(path)):
#     print(num)


# path = 'AR_aligment_initial/dbf7/'
# for filename in os.listdir(path):

#         image = cv2.imread(path + filename)
#         print(image.shape)
#         cv2.imwrite(f'AR_aligment_final/ID49/{filename}', image)

# number = [0 for i in range(90)]
# path = 'AR_aligment_final/'
# for index, id in enumerate(os.listdir(path)):
#     print(id)
#     for filename in os.listdir(path + id):
#         number[index] += 1
#
# print(number)

# name = []
# path = f'AR_aligment_final/ID'
#
# for index in range(1, 91):
#     path = f'AR_aligment_final/ID{index}/'
#     for num, file in enumerate(os.listdir(path)):
#         if num == 1:
#             break
#         name.append(file[0:5])
#
#
# path = 'AR_original_data/'
# for file in os.listdir(path):
#     for filename in os.listdir(path + file):
#         if 'bmp' not in filename:
#             continue
#         image = cv2.imread(path + file + '/' + filename)
#         if filename[0:5] in name:
#             cv2.imwrite(f'AR_original_data_clf/ID{name.index(filename[0:5]) + 1}/{filename}', image)

# path = 'AR_original_data_clf/'
# for id in os.listdir(path):
#     for filename in os.listdir(path + id):
#         os.remove(path + id + '/' + filename)

# source_path = 'AR_original_alignment/'
# restore_path = 'AR_original_alignment_test21/'
# file_id = {'m-010': 'ID1', 'm-020': 'ID2', 'm-021': 'ID3', 'm-025': 'ID4', 'm-031': 'ID5', 'm-041': 'ID6', 'm-054': 'ID7',
#            'm-056': 'ID8', 'm-066': 'ID9', 'w-002': 'ID10', 'w-003': 'ID11', 'w-010': 'ID12', 'w-018': 'ID13', 'w-019': 'ID14',
#            'w-025': 'ID15', 'w-039': 'ID16', 'w-043': 'ID17', 'w-044': 'ID18', 'w-048': 'ID19', 'w-053': 'ID20', 'w-055': 'ID21'}


# for i in range(1, 22):
#     os.makedirs(restore_path + f'ID{i}')
# for ID in os.listdir(restore_path):
#     for filename in os.listdir(restore_path + ID):
#         os.remove(restore_path + ID + '/' + filename)

# for folder in os.listdir(source_path):
#     print(folder)
#     for filename in os.listdir(source_path + folder):
#         if 'bmp' not in filename:
#             continue
#         if filename[0: 5] in file_id:
#             image = cv2.imread(source_path + folder + '/' + filename)
#             cv2.imwrite(restore_path + file_id[filename[0: 5]] + '/' + filename, image)




