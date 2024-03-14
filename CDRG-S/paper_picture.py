from regression_test import *
import cv2
import numpy as np


def data_vis(path_num=1):
    if path_num == 1:
        path = '/disk2/bosen/Datasets/AR_train/'
        ID = ['ID1/', 'ID2/', 'ID3/', 'ID89/', 'ID88/']
    if path_num == 2:
        path = '/disk2/bosen/Datasets/AR_aligment_other/'
        ID = ['ID01/', 'ID02/', 'ID03/', 'ID89/', 'ID88/']
    if path_num == 3:
        path = '/disk2/bosen/Datasets/AR_aug100_rank6_10_train/'
        ID = ['ID01/', 'ID02/', 'ID03/', 'ID89/', 'ID88/']
    if path_num == 4:
        path = '/disk2/bosen/Datasets/AR_aug100_rank3_5_train/'
        ID = ['ID01/', 'ID02/', 'ID03/', 'ID89/', 'ID88/']
    if path_num == 5:
        path = '/disk2/bosen/Datasets/AR_test/'
        ID = ['ID01/', 'ID02/', 'ID03/', 'ID20/', 'ID21/']


    vis_data = [[] for i in range(5)]
    for id_num, id in enumerate(ID):
        for num, filename in enumerate(os.listdir(path + id)):
            if num == 20:
                break
            image = cv2.imread(path + id + filename, 0) / 255
            image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
            vis_data[id_num].append(image)

    plt.subplots(figsize=(5, 3))
    plt.subplots_adjust(hspace=0, wspace=0)
    for i in range(5):
        std_dev = np.std(vis_data[i], axis=0)
        mean = np.mean(vis_data[i], axis=0)

        plt.subplot(3, 5, i+1)
        plt.axis('off')
        plt.imshow(vis_data[i][0], cmap='gray')
        plt.subplot(3, 5, i+6)
        plt.axis('off')
        plt.imshow(mean, cmap='gray')
        plt.subplot(3, 5, i+11)
        plt.axis('off')
        plt.imshow(std_dev, cmap='hot')

    plt.show()

for i in range(1, 6):
    data_vis(i)