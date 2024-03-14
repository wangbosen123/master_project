from overall_model import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import cv2
import seaborn as sns

def deal_cls_data():
    train_path = 'cls_data/train_data/'
    test_var_large_path = 'cls_data/test_data_var_large/'
    test_var_less_path = 'cls_data/test_data_var_less/'

    AR_90id_path = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_train90/"
    AR_21id_path = "/home/bosen/gradation_thesis/AR_original_data_aligment/AR_original_alignment_test21/"

    ID = [f'ID{i}' for i in range(1, 112)]
    for id in ID:
        os.mkdir(train_path + id)
        os.mkdir(test_var_large_path + id)
        os.mkdir(test_var_less_path + id)

    for id in os.listdir(AR_90id_path):
        for filename in os.listdir(AR_90id_path + id):
            image = cv2.imread(AR_90id_path + id + '/' + filename)
            if '-1-0' in filename:
                cv2.imwrite(train_path + id + '/' + filename, image)
            if '-1-1' in filename or '-1-2' in filename:
                cv2.imwrite(test_var_less_path + id + '/' + filename, image)
            if '-3-0' in filename or '-3-1' in filename:
                cv2.imwrite(test_var_large_path + id + '/' + filename, image)

    for id in os.listdir(AR_21id_path):
        for filename in os.listdir(AR_21id_path + id):
            image = cv2.imread(AR_21id_path + id + '/' + filename)
            if '-1-0' in filename:
                cv2.imwrite(train_path + f'ID{int(id[2:])+90}' + '/' + filename, image)
            if '-1-1' in filename or '-1-2' in filename:
                cv2.imwrite(test_var_less_path + f'ID{int(id[2:])+90}' + '/' + filename, image)
            if '-3-0' in filename or '-3-1' in filename:
                cv2.imwrite(test_var_large_path + f'ID{int(id[2:])+90}' + '/' + filename, image)


def cls_train():
    train_path = "/home/bosen/gradation_thesis/synthesis_system/cls_datasets/train_data_var_less/"
    test_path = 'cls_data/test_data_var_less/'
    test_path = "/home/bosen/gradation_thesis/synthesis_system/cls_datasets/train_data_var_large/"


    train_image, train_label = [], []
    test_image, test_label = [], []
    for id in os.listdir(test_path):
        for filename in os.listdir(test_path + id):
            if 'bmp' in filename:
                continue
            image = cv2.imread(test_path + id + '/' + filename, 0) / 255
            test_image.append(image)
            test_label.append(tf.one_hot(int(id[2:])-1, 111))
    test_image, test_label = np.array(test_image).reshape(-1, 64, 64, 1), np.array(test_label)


    for id in os.listdir(train_path):
        for filename in os.listdir(train_path + id):
            image = cv2.imread(train_path + id + '/' + filename, 0) / 255
            train_image.append(image)
            train_label.append(tf.one_hot(int(id[2:])-1, 111))

    train_image, train_label = np.array(train_image).reshape(-1, 64, 64, 1), np.array(train_label)
    data = list(zip(train_image, train_label))
    np.random.shuffle(data)
    data = list(zip(*data))
    train_image, train_label = np.array(data[0]).reshape(-1, 64, 64, 1), np.array(data[1])
    print(train_image.shape, train_label.shape)
    print(np.argmax(train_label, axis=-1))


    input = Input((64, 64, 1))
    out = Conv2D(32, 3, activation=LeakyReLU(0.3), strides=(1, 1), padding="same")(input)
    out = Conv2D(32, 3, activation=LeakyReLU(0.3), strides=(1, 1), padding="same")(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((2, 2))(out)

    out = Conv2D(64, 3, activation=LeakyReLU(0.3), strides=(1, 1), padding="same")(out)
    out = Conv2D(64, 3, activation=LeakyReLU(0.3), strides=(1, 1), padding="same")(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((2, 2))(out)

    out = Conv2D(128, 3, activation=LeakyReLU(0.3), strides=(1, 1), padding="same")(out)
    out = Conv2D(128, 3, activation=LeakyReLU(0.3), strides=(1, 1), padding="same")(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((2, 2))(out)

    out = Flatten()(out)
    out = Dense(512, activation=LeakyReLU(0.4))(out)
    out = Dropout(0.4)(out)
    out = Dense(256, activation=LeakyReLU(0.4))(out)
    out = Dropout(0.4)(out)
    out = Dense(111, activation='softmax')(out)

    model = Model(input, out)
    model.compile(optimizer=tf.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_image, train_label, batch_size=6, epochs=12, verbose=1, validation_data=(test_image, test_label))
    model.save('model_weight/normal_cls3.h5')
    return model

def confu_matrix():
    model = load_model('model_weight/normal_cls3.h5')
    test32_image, test32_label, test32_pred = [], [], []
    test20_image, test20_label, test20_pred = [], [], []
    test16_image, test16_label, test16_pred = [], [], []
    test12_image, test12_label, test12_pred = [], [], []
    test8_image, test8_label, test8_pred = [], [], []

    test_path = 'cls_data/test_data_var_less/'
    # test_path = "/home/bosen/gradation_thesis/synthesis_system/cls_datasets/train_data_var_large/"

    for id in os.listdir(test_path):
        for filename in os.listdir(test_path + id):
            if 'bmp' in filename:
                continue
            image = cv2.imread(test_path + id + '/' + filename, 0) / 255
            pred = model(image.reshape(1, 64, 64, 1))

            if '32' in filename:
                test32_pred.append(tf.reshape(pred, [111]))
                test32_label.append(int(id[2:])-1)
            if '20' in filename:
                test20_pred.append(tf.reshape(pred, [111]))
                test20_label.append(int(id[2:])-1)
            if '16' in filename:
                test16_pred.append(tf.reshape(pred, [111]))
                test16_label.append(int(id[2:])-1)
            if '12' in filename:
                test12_pred.append(tf.reshape(pred, [111]))
                test12_label.append(int(id[2:])-1)
            if '8' in filename:
                test8_pred.append(tf.reshape(pred, [111]))
                test8_label.append(int(id[2:])-1)
    print(accuracy_score(test32_label, np.argmax(test32_pred, axis=-1)))
    print(accuracy_score(test20_label, np.argmax(test20_pred, axis=-1)))
    print(accuracy_score(test16_label, np.argmax(test16_pred, axis=-1)))
    print(accuracy_score(test12_label, np.argmax(test12_pred, axis=-1)))
    print(accuracy_score(test8_label, np.argmax(test8_pred, axis=-1)))
    cf32 = sns.heatmap(confusion_matrix(test32_label, np.argmax(test32_pred, axis=-1)), vmin=0, vmax=1)
    plt.show()
    cf20 = sns.heatmap(confusion_matrix(test20_label, np.argmax(test20_pred, axis=-1)), vmin=0, vmax=1)
    plt.show()
    cf16 = sns.heatmap(confusion_matrix(test16_label, np.argmax(test16_pred, axis=-1)), vmin=0, vmax=1)
    plt.show()
    cf12 = sns.heatmap(confusion_matrix(test12_label, np.argmax(test12_pred, axis=-1)), vmin=0, vmax=1)
    plt.show()
    cf8 = sns.heatmap(confusion_matrix(test8_label, np.argmax(test8_pred, axis=-1)), vmin=0, vmax=1)
    plt.show()


def distillation_loss(zd, zg):
    dot_product_d_space = tf.matmul(zd, tf.transpose(zd))
    dot_product_g_space = tf.matmul(zg, tf.transpose(zg))
    square_norm_d_space = tf.linalg.diag_part(dot_product_d_space)
    square_norm_g_space = tf.linalg.diag_part(dot_product_g_space)

    distances_d_space = tf.sqrt(tf.expand_dims(square_norm_d_space, 1) - 2.0 * dot_product_d_space + tf.expand_dims(square_norm_d_space, 0) + 1e-8)
    distances_g_space = tf.sqrt(tf.expand_dims(square_norm_g_space, 1) - 2.0 * dot_product_g_space + tf.expand_dims(square_norm_g_space, 0) + 1e-8)
    print(distances_d_space, distances_g_space)

    norm_distances_d_space = distances_d_space / (tf.reduce_sum(distances_d_space / 2))
    norm_distances_g_space = distances_g_space / (tf.reduce_sum(distances_g_space / 2))
    distance = tf.math.abs(norm_distances_d_space - norm_distances_g_space)
    distance = tf.reduce_sum(distance) / 2
    return distance



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

for i in range(100):
    print(i)

