import matplotlib.pyplot as plt
from load_data_ck import *
from loss_ck import *
from build_model_ck import *
import tensorflow as tf
from tensorflow.keras.models import *
import time


def train_step(nocc_image, occ_image):
    input = tf.concat([nocc_image, occ_image], axis=-1)
    with tf.GradientTape() as tape:
        _,feature1,feature2,_,_ = model(input)
        diff_loss = different_loss(feature1, feature2)
    grads = tape.gradient(diff_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return diff_loss

def training(epochs, batch_num, batch_size):
    diff_loss = []
    diff_loss_avg= []
    for epoch in range(epochs):
        start = time.time()
        train_path, train_label = load_path()
        train_nocc, train_occ = load_image(train_path)
        for batch in range(batch_num):
            nocc, occ, label = get_batch_data(train_nocc, train_occ, train_label, batch, batch_size)
            different_loss = train_step(nocc, occ)
            diff_loss.append(different_loss)
        diff_loss_avg.append(np.mean(different_loss))
        print("___________________________")
        print(f"the epcoh is {epoch+1}")
        print(f"the different_loss is {diff_loss_avg[-1]}")
        print("the spent time is %s" %(time.time() - start))
        if epoch > 93 and diff_loss_avg[-1] == np.min(diff_loss_avg):
            model.save_weights(f"model_weight_ck/pdsn_mask_{epoch+1}_weights")

    return diff_loss_avg



if __name__ =="__main__":
    model = PDSN_model()
    model.load_weights("model_weight_ck/pdsn_cls_100_weights")
    optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range(12):
        if i != 4 and i != 5:
            model.layers[i].trainable = False

    diff_loss = training(100,20,42)

    plt.plot(diff_loss)
    plt.title("the different loss ")
    plt.savefig("loss_image_ck/pdsn_maks_different_loss.jpg")
    plt.close()
