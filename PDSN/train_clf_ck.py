from load_data_ck import *
from loss_ck import *
from build_model_ck import *
import time
from sklearn.metrics import accuracy_score

def train_step(nocc_image, occ_image, label):
    input = tf.concat([nocc_image,occ_image],axis=-1)
    with tf.GradientTape() as tape:
        _, _, _, pred, _ = model(input)
        clf_loss = cls_loss(label, pred)
    grads = tape.gradient(clf_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return clf_loss , pred

def training(epochs,batch_num,batch_size):
    cls_loss = []
    cls_loss_avg = []
    accuracy = []
    accuracy_avg = []
    for epoch in range(epochs):
        start = time.time()
        train_path , train_label = load_path()
        train_nocc, train_occ = load_image(train_path)
        for batch in range(batch_num):
            nocc , occ, label = get_batch_data(train_nocc, train_occ, train_label, batch, batch_size)
            clf_loss , pred = train_step(nocc, occ, label)
            pred = tf.argmax(pred,axis=-1)
            label = tf.argmax(label,axis=-1)
            cls_loss.append(clf_loss)
            accuracy.append(accuracy_score(label, pred))
        cls_loss_avg.append(np.mean(clf_loss))
        accuracy_avg.append(np.mean(accuracy))
        print("________________________________")
        print(f"the epoch is {epoch+1}")
        print(f"the cls_loss is {cls_loss_avg[-1]}")
        print(f"the accuracy is {accuracy_avg[-1]}")
        print(f"the spend times is %s" %(time.time() - start))
        if accuracy_avg[-1] > 0.94:
            model.save_weights(f"model_weight_ck/pdsn_cls_{epoch+1}_weights")



if __name__ == "__main__":
    optimizer = tf.keras.optimizers.Adam(1e-5)
    model = PDSN_model()
    training(100,20,42)