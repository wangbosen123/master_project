from load_data import *
from loss import *
from build_model import *
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
        x_train, x_test, y_train, y_test = load_mnist()
        train_nocc, train_occ, train_label = prepare_data(x_train, y_train)
        for batch in range(batch_num):
            nocc , occ , label = batch_data(train_nocc, train_occ , train_label, batch, batch_size)
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
        if accuracy_avg[-1] > 0.95:
            model.save_weights(f"model_weight/pdsn_cls_{epoch+1}_weights")
    return cls_loss_avg , accuracy_avg



if __name__ == "__main__":
    optimizer = tf.keras.optimizers.Adam(5e-6)
    model = PDSN_model()
    training(140,1000,60)

