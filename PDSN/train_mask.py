import matplotlib.pyplot as plt
from load_data import *
from loss import *
from build_model import *
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
        x_train, x_test , y_train, y_test = load_mnist()
        train_nocc, train_occ, train_label = prepare_data(x_train, y_train)
        for batch in range(batch_num):
            nocc , occ , label = batch_data(train_nocc, train_occ , train_label, batch, batch_size)
            different_loss = train_step(nocc, occ)
            diff_loss.append(different_loss)
        diff_loss_avg.append(np.mean(different_loss))
        print("___________________________")
        print(f"the epcoh is {epoch+1}")
        print(f"the different_loss is {diff_loss_avg[-1]}")
        print("the spent time is %s" %(time.time() - start))
        if epoch == 49:
            model.save_weights(f"model_weight/pdsn_mask_{epoch+1}_weights")

    return diff_loss_avg



if __name__ =="__main__":
    model = PDSN_model()
    model.load_weights("model_weight/pdsn_cls_85_weights")
    optimizer = tf.keras.optimizers.Adam(1e-6)
    for i in range(4):
        model.layers[i].trainable=False

    diff_loss = training(50,1000,60)

    plt.plot(diff_loss)
    plt.title("the different loss ")
    plt.savefig("loss_img/pdsn_maks_different_loss.jpg")
    plt.close()
