import cv2
import build_model
from prepare_data import *
from build_model import *
from sklearn.decomposition import PCA
import csv
from relation_model import *

def test_model():
    image = cv2.imread("target_high_blur.jpg", cv2.IMREAD_GRAYSCALE)
    image = image.reshape(1, 64, 64, 1) / 255
    code = encoder(image)
    gen_image = generator(code)
    gen_image = tf.reshape(gen_image, [64,64])
    plt.imshow(gen_image, cmap="gray")
    plt.savefig("gen_image.jpg")

def diversity(low=True):
    train_data = []
    idx_1 = 0
    idx_2 = 0
    for filename in os.listdir("celeba_test"):
        image = cv2.imread("celeba_test"+"/"+filename)
        if "6708" in filename:
            test_1 = idx_1
        if "6591" in filename:
            test_2 = idx_2
        idx_1 += 1
        idx_2 += 1
        if low:
            image = cv2.GaussianBlur(image, (5, 5), sigmaX=1, sigmaY=1)
            image = cv2.resize(image, (8, 8), cv2.INTER_CUBIC)
            image= cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        train_data.append(image)
    train_data = np.array(train_data)
    train_data = train_data.reshape(-1,64,64,3)/255

    feature_extraction = tf.keras.applications.vgg16.VGG16(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
    feature = feature_extraction(train_data)
    feature = tf.reshape(feature,[1000,2048]).numpy()
    pca = PCA(n_components=3)
    new_feature = pca.fit_transform(feature)


    #X axis new_feature[0:,0:1] , Y axis new_feature[0:,1:]
    fig = plt.figure()
    ax3 = plt.axes(projection="3d")

    ax3.scatter(new_feature[test_1:test_1 + 1, 0:1], new_feature[test_1:test_1 + 1, 1:2], new_feature[test_1:test_1 + 1, 2:])
    ax3.scatter(new_feature[test_2:test_2 + 1, 0:1], new_feature[test_2:test_2 + 1, 1:2], new_feature[test_2:test_2 + 1, 2:])
    # ax3.scatter(new_feature[0:,0:1], new_feature[0:,1:2], new_feature[0:,2:])
    plt.savefig("diversity")
    plt.close()


#obtain all train_data latent code
def get_train_low_code():
    train_data_high = []
    train_data_low = []
    gen_image = []
    latent_code = []
    for filename in os.listdir("celeba_train"):
        image = cv2.imread("celeba_train"+"/"+filename, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        blur = cv2.GaussianBlur(image, (5, 5), sigmaX=1, sigmaY=1)
        low_img = cv2.resize(blur, (8, 8), cv2.INTER_CUBIC)
        low_img = cv2.resize(low_img, (64, 64), cv2.INTER_CUBIC)
        train_data_high.append(image)
        train_data_low.append(low_img)
        image = image.reshape(1,64,64,1)
        code = encoder(image)
        latent_code.append(code)
        gen_img = generator(code)
        gen_img = tf.reshape(gen_img, [64,64])
        gen_image.append(gen_img)

    train_data_high, train_data_low, latent_code, gen_image = np.array(train_data_high), np.array(train_data_low), np.array(latent_code), np.array(gen_image)
    plt.subplots(figsize=(15,4))
    for i in range(10):
        plt.subplot(3,10,i+1)
        plt.axis("off")
        plt.imshow(train_data_low[i], cmap="gray")
        plt.subplot(3,10,i+11)
        plt.axis("off")
        plt.imshow(gen_image[i],cmap="gray")
        plt.subplot(3, 10, i + 21)
        plt.axis("off")
        plt.imshow(train_data_high[i],cmap="gray")
    plt.savefig("result_image")




if __name__ == "__main__":
    encoder = encoder()
    generator = generator()
    # discriminator = discriminator()
    #
    #set the checkpoint
    checkpoint_encoder = tf.train.Checkpoint(encoder)
    checkpoint_generator = tf.train.Checkpoint(generator)
    # checkpoint_discriminator = tf.train.Checkpoint(discriminator)
    checkpoint_encoder.restore("highface_model_weight/encoder/encoder-1040")
    checkpoint_generator.restore("highface_model_weight/generator/generator-1040")
    # checkpoint_discriminator.restore("highface_model_weight/discriminator/discriminator-1000")
    #
    # train_path = load_train_path()
    # train_path.sort()
    # print(train_path[1])

    # target = cv2.imread("target_low.jpg", cv2.IMREAD_GRAYSCALE)
    # target = target.reshape(1,64,64,1)/255
    # code = encoder(target)
    # img = generator(code)
    # img = tf.reshape(img, [64,64])
    # plt.imshow(img,cmap="gray")
    # plt.savefig("enocder_init.jpg")

    # real_high, real_low = load_image(get_batch_data(train_path, 0, 5), path="celeba_train")
    # original_code = encoder(real_low)
    # np.savetxt('sample.csv', original_code, delimiter=",")
    # original_fake_high = generator(original_code)
    # original_fake_high = tf.reshape(original_fake_high, [-1, 64, 64])
    # plt.imshow(original_fake_high[1],cmap="gray")
    # plt.savefig("enocder_init.jpg")

    # code = []
    # with open('sample.csv', newline='') as csvfile:
    #     rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    #     for row in rows:
    #         code.append(row)
    # code = np.array(code)
    #
    # # print(type(code[0][0]))
    # code = code[0].reshape(1,200)
    # img = generator(code)
    # print(img.shape)
    # img = tf.reshape(img,[64,64])
    # plt.imshow(img, cmap="gray")
    # plt.savefig("result_image")


    # train = load_train_path()
    # real_high, real_low = load_image(get_batch_data(train,1,300),path="celeba_train")
    # real_low = real_low.reshape(-1,64,64,1)
    # code = encoder(real_low)
    # gen = generator(code)
    # gen = tf.reshape(gen, [-1, 64, 64])
    #
    # code = []
    # with open('inverted_code/1_batch.csv', newline='') as csvfile:
    #     rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    #     for row in rows:
    #         code.append(row)
    # code = np.array(code)
    # code = code[0:30]
    # update = generator(code)
    # update = tf.reshape(update, [-1, 64, 64])
    #
    # plt.subplots(figsize=(15,4))
    # for i in range(5):
    #     plt.subplot(2,5,i+1)
    #     plt.axis("off")
    #     plt.imshow(gen[i+10], cmap="gray")
    #     plt.subplot(2, 5, i + 6)
    #     plt.axis("off")
    #     plt.imshow(update[i+10], cmap="gray")
    # plt.savefig("test.jpg")


    # validate the init code inver code relation code
    inv_code = []
    model = load_model("highface_model_weight/relation_model")
    ground_truth = cv2.imread("part of celebA/040087.jpg", cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(ground_truth, (5, 5), sigmaX=1, sigmaY=1)
    low_img = cv2.resize(blur, (8, 8), cv2.INTER_CUBIC)
    target = cv2.resize(low_img, (64, 64), cv2.INTER_CUBIC)
    target = target.reshape(1,64,64,1)/255
    init_code = encoder(target)
    update_code = model(init_code)
    init_code = np.array(init_code[0])
    update_code = np.array(update_code[0])

    with open(f'inverted_code/single_inv.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in rows:
            inv_code.append(row)
    # print(inv_code[0])
    # print(init_code)
    # print(update_code)
    # plt.plot(init_code)
    # plt.savefig("the init.jpg")
    # plt.plot(update_code)
    # plt.savefig("the update.jpg")
    plt.plot(inv_code[0])
    plt.savefig("the inv.jpg")
    # plt.legend(["init", "update", "inv_code"], loc="upper right")
    plt.savefig("the relation.jpg")
    # gen_img = generator(update_code)
    # gen_img = tf.reshape(gen_img, [64,64])
    # gen_img = np.array(gen_img)*255
    # cv2.imwrite("result.jpg",gen_img)

