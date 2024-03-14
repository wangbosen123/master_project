import cv2

import build_model
from prepare_data import *
from build_model import *
import matplotlib.cm as cm


last_conv_layer_name = "conv1"
network_layer_name = ["drop1", "conv2", "drop2", "flat", "output"]


def gradcam_heatmap(img_array, model, last_conv_layer_name, network_layer_name):
    pred = []
    # 建一個從頭到最後的conv layer 的模型
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # 建一個從 conv後面到最後分類 的模型
    network_input = Input(shape=last_conv_layer.output.shape[1:])
    x = network_input
    for layer_name in network_layer_name:
        x = model.get_layer(layer_name)(x)
    network_model = Model(network_input, x)

    with tf.GradientTape() as tape:
        # 將img_array 帶入 last_conv_layer_model 得到每一個輸出的pixel value be a output
        last_conv_layer_output = last_conv_layer_model(img_array)
        # 用watch 找完gradient 後自動帶入 last_conv_layer_output
        tape.watch(last_conv_layer_output)
        # preds 為 last_conv_layer_output 帶入network_model 等同於將img_array 做分類
        preds = network_model(last_conv_layer_output)
        print(preds)
        pred.append(preds[0])
        # preds 為 [[.... 分類數]] top_pred_index 為預測結果ex:第一類
        pred_index = tf.argmax(preds[0])
        # Yc 對此偏為微分
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    # grads.shape = (28,28,30) 透過reduce_mean 可以將pooled_grads 變成30張 每張為28*28的平均
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 將last_conv_layer_output 傳承numpy type
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    # 建一個28*28 shape 的heatmap 裝載加總的 權重*特徵圖
    heatmap = np.zeros((32, 32))

    # 將每一個特徵圖*對應的pooled_grads[i], 且加起來得到heatmap
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
        heatmap += last_conv_layer_output[:, :, i]

    # 將加總的 heatmap 做relu 且正規至0~1
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = np.uint8(255 * heatmap)

    # 用 jet 畫圖
    cmap = cm.get_cmap("jet")
    cmap_colors = cmap(np.arange(256))[:, :3]
    cmap_heatmap = cmap_colors[heatmap]
    return cmap_heatmap, pred




def draw_heatmap():
    real_map, fake_map = [], []
    real_d, fake_d = [], []
    last_conv_layer_name = "conv1"
    network_layer_name = ["drop1", "conv2", "drop2", "flat", "output"]
    train, test = load_path()
    real_low, real_high = load_image(get_batch_data(train, 6, 100), train=True)
    code = encoder(real_low)
    fake_img = generator(code)

    for i in range(100):
        real = real_high[i].reshape(1,64,64,1)
        fake = tf.reshape(fake_img[i],[1,64,64,1])
        print("___________________________________________________")
        real_heatmap, real_pred = gradcam_heatmap(real, dis_encoder, last_conv_layer_name, network_layer_name)
        fake_heatmap, fake_pred = gradcam_heatmap(fake, dis_encoder, last_conv_layer_name, network_layer_name)

        real_map.append(real_heatmap)
        fake_map.append(fake_heatmap)
        real_d.append(np.array(real_pred)[0][0])
        fake_d.append(np.array(fake_pred)[0][0])

    fake_img = tf.reshape(fake_img,[-1,64,64])
    plt.plot(real_d)
    plt.plot(fake_d)
    plt.title("Real and Gen_img Prediction")
    plt.legend(["Real_Prediction", "Fake_Prediction"], loc="upper right")
    plt.savefig("result/visualized/Real and Gen_img Prediction")
    plt.close()

    # plt.subplots(figsize=(10,4))
    # for i in range(10):
    #     plt.subplot(4,10,i+1)
    #     plt.axis("off")
    #     plt.imshow(real_high[i], cmap="gray")
    #     plt.subplot(4, 10, i + 11)
    #     plt.axis("off")
    #     plt.imshow(real_map[i])
    #     plt.subplot(4,10,i+21)
    #     plt.axis("off")
    #     plt.imshow(fake_img[i], cmap="gray")
    #     plt.subplot(4,10,i+31)
    #     plt.axis("off")
    #     plt.imshow(fake_map[i])
    # plt.savefig("result/visualized/discriminator_encoder_heatmap")
    # plt.close()




if __name__ == "__main__":
    encoder = encoder()
    generator = generator()
    dis_encoder = discriminator()
    dis_encoder.load_weights("model_weight/instance_discriminator/discriminator-500")
    encoder.load_weights("model_weight/instance_encoder/encoder-501")
    generator.load_weights("model_weight/instance_generator/generator-501")
    draw_heatmap()
    # last_conv_layer_name = "conv1"
    # network_layer_name = ["drop1", "conv2", "drop2", "flat", "output"]
    # image = cv2.imread("celeba_train/0_train.jpg", cv2.IMREAD_GRAYSCALE)
    # image = image.reshape(1,64,64,1) / 255
    # map = gradcam_heatmap(image, dis_encoder, last_conv_layer_name, network_layer_name)
    # map = map*255
    # print(map)
    # cv2.imwrite("result/visualized/single.jpg", map)
