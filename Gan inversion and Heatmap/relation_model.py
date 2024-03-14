import cv2

from prepare_data import *
from build_model import *
from highfaceGAN import *
from all_inversion import *
import csv

def get_train_low_code():
    latent_code = []
    path = load_path(train=True)
    real_high, real_low = load_image(path, path="part of celebA")
    for img in real_low:
        img = img.reshape(1,64,64,1)
        code = encoder(img)
        code = tf.reshape(code,[200])
        latent_code.append(code)
    latent_code = np.array(latent_code)
    return latent_code

def get_update_code():
    code = []
    for i in range(15):
        with open(f'inverted_code/train_{i}_batch.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in rows:
                code.append(row)
    code = np.array(code)
    return code

def inference(train=True):
    model = load_model("highface_model_weight/relation_model")
    if train:
        train_path = load_path(train=True)
        real_high, real_low = load_image(get_batch_data(train_path, 10, 400), path="part of celebA")
        real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)
    else:
        test_path = load_path(train=False)
        real_high, real_low = load_image(get_batch_data(test_path, 1, 400), path="part of celebA")
        real_high, real_low = real_high.reshape(-1, 64, 64, 1), real_low.reshape(-1, 64, 64, 1)

    init_code = encoder(real_low)
    init_img = generator(init_code)
    init_img = tf.reshape(init_img, [-1,64,64])

    update_code = model(init_code)
    update_img = generator(update_code)
    update_img = tf.reshape(update_img, [-1,64,64])

    plt.subplots(figsize=(6, 5))
    for i in range(5):
        plt.subplot(4, 5, i + 1)
        plt.axis("off")
        plt.imshow(real_high[i], cmap="gray")
        plt.subplot(4, 5, i + 6)
        plt.axis("off")
        plt.imshow(real_low[i], cmap="gray")
        plt.subplot(4, 5, i + 11)
        plt.axis("off")
        plt.imshow(init_img[i], cmap="gray")
        plt.subplot(4, 5, i + 16)
        plt.axis("off")
        plt.imshow(update_img[i], cmap="gray")

    plt.savefig(f"relation_{train}.jpg")
    plt.close()





if __name__ == "__main__":
    relation_model = mapping_network()
    encoder = build_model.encoder()
    generator = build_model.generator()

    # set checkpoint
    checkpoint_encoder = tf.train.Checkpoint(encoder)
    checkpoint_generator = tf.train.Checkpoint(generator)
    checkpoint_encoder.restore("highface_model_weight/encoder/encoder-1040")
    checkpoint_generator.restore("highface_model_weight/generator/generator-1040")

    # train the mapping network
    init_code = get_train_low_code()
    update_code = get_update_code()
    history = relation_model.fit(init_code, update_code, epochs=200, batch_size=30,verbose=1, validation_split=0.1)

    loss = history.history['loss']
    loss_val = history.history["val_loss"]
    plt.plot(loss)
    plt.plot(loss_val)
    plt.legend(["loss_train", "loss_val"], loc="upper right")
    plt.title("the mse loss")
    plt.savefig("the mse loss")
    relation_model.save("highface_model_weight/relation_model")


    # #inference stage
    inference(train=True)
    inference(train=False)



    # check whether the init and update code match or not ?
    # init_code = get_train_low_code()
    # update_code = get_update_code()
    # print(init_code.shape)
    # init_code = init_code[0].reshape(1, 200)
    # update_code = update_code[0].reshape(1, 200)
    # print(update_code.shape)
    # print(init_code.shape)
    # init = generator(init_code)
    # update = generator(update_code)
    # init, update = tf.reshape(init, [64, 64]), tf.reshape(update, [64, 64])
    # init, update = np.array(init) * 255, np.array(update) * 255
    # cv2.imwrite("result1.jpg", init)
    # cv2.imwrite("result2.jpg", update)