from load_data import *
from loss import *
from build_model import *
from sklearn.metrics import accuracy_score

def test(train="clf"):
    train_path, label = load_path()
    train_nocc, train_occ = load_image(train_path)
    model = PDSN_model()

    for i in range(200,700,25):
        nocc ,occ = train_nocc[i].reshape(1,128,128,1), train_occ[i].reshape(1,128,128,1)
        input = tf.concat([nocc,occ],axis=-1)
        if train == "clf":
            model.load_weights("model_weight/pdsn_cls_85_weights")
            _,_,_,pred,_ = model(input)
        if train == "total":
            model.load_weights("model_weight/PDSN_total_train_60")
            _,_,_,_,pred = model(input)

        print("pred" , tf.argmax(pred,axis=-1))
        print("label" , label[i])

def test_accuracy():
    test_path , label = load_path(train=False)
    test_nocc, test_occ = load_image(test_path)
    model = PDSN_model()
    model.load_weights("model_weight/PDSN_total_train_60")
    nocc , occ = test_nocc.reshape(-1,128,128,1), test_occ.reshape(-1,128,128,1)
    input = tf.concat([nocc, occ], axis=-1)
    _, _, _, _, pred = model(input)
    pred = tf.argmax(pred, axis=-1)
    accuracy = accuracy_score(label , pred)

    return accuracy


if __name__ == "__main__":
    test(train="total")
    accuracy = test_accuracy()
    print(accuracy)
