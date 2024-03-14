import matplotlib.pyplot as plt
from  load_data_ck import *
from loss_ck import *
from build_model_ck import *
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score



def train_step(nocc_image , occ_image , label):
    input = tf.concat([nocc_image, occ_image], axis=-1)
    with tf.GradientTape() as tape:
        _,feature1,feature2,_,pred_for_occ = model(input)
        diff_loss = different_loss(feature1,feature2)
        clf_loss = cls_loss(label,pred_for_occ)
        pred_for_occ = tf.argmax(pred_for_occ, axis=-1)
        label = tf.argmax(label, axis=-1)
        accuracy = accuracy_score(label, pred_for_occ)
        total_loss = diff_loss + clf_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return diff_loss , clf_loss , accuracy

def training(epochs, batch_num, batch_size):
    diff_loss = []
    diff_loss_avg = []
    cls_loss = []
    cls_loss_avg = []
    accuracy = []
    accuracy_avg = []
    for epoch in range(epochs):
        start = time.time()
        train_path, train_label = load_path()
        train_nocc, train_occ = load_image(train_path)
        for batch in range(batch_num):
            nocc, occ, label = get_batch_data(train_nocc, train_occ, train_label, batch, batch_size)
            different_loss , classfy_loss ,acc = train_step(nocc, occ , label)
            diff_loss.append(different_loss)
            cls_loss.append(classfy_loss)
            accuracy.append(acc)
        diff_loss_avg.append(np.mean(different_loss))
        cls_loss_avg.append(np.mean(cls_loss))
        accuracy_avg.append(np.mean(accuracy))
        print("__________________________________________")
        print(f"the epoch is {epoch+1}")
        print(f"the different_loss is : {diff_loss_avg[-1]}")
        print(f"the cls_loss is : {cls_loss_avg[-1]}")
        print(f"the accuracy is {accuracy_avg[-1]}")
        print("the spend time is %s" %(time.time() - start ))

        if epoch ==199:
            model.save_weights(f"model_weight_ck/PDSN_total_train_{epoch+1}")


    return diff_loss_avg, cls_loss_avg, accuracy_avg





if __name__ == "__main__":
    model = PDSN_model()
    model.load_weights("model_weight_ck/pdsn_mask_100_weights")
    optimizer = tf.keras.optimizers.Adam(1e-5)
    for layer in model.layers:
        print(layer, layer.trainable)

    diff_loss , clf_loss, accuracy = training(200,20,42)

    plt.plot(diff_loss)
    plt.title("the different loss ")
    plt.savefig(f"loss_image_ck/the total_train_different_loss.jpg")
    plt.close()

    plt.plot(clf_loss)
    plt.title("the cls loss ")
    plt.savefig(f"loss_image_ck/the total_train_cls_loss.jpg")
    plt.close()

    plt.plot(accuracy)
    plt.title("the accuracy ")
    plt.savefig("loss_image_ck/the total_train_accuracy.jpg")
    plt.close()