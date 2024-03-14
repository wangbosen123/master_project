from load_data_ck import *
from loss_ck import *
from build_model_ck import *
from sklearn.metrics import accuracy_score

def test(train="clf"):
    train_path, label = load_path()
    train_nocc, train_occ = load_image(train_path)
    model = PDSN_model()

    for i in range(200,700,25):
        nocc ,occ = train_nocc[i].reshape(1,128,128,1), train_occ[i].reshape(1,128,128,1)
        input = tf.concat([nocc,occ],axis=-1)
        if train == "clf":
            model.load_weights("model_weight_ck/pdsn_cls_100_weights")
            _,_,_,pred,_ = model(input)
        if train == "total":
            model.load_weights("model_weight_ck/PDSN_total_train_200")
            _,_,_,_,pred = model(input)

        print("pred" , tf.argmax(pred,axis=-1))
        print("label" , tf.argmax(label[i],axis=-1))

def test_accuracy():
    test_path , label = load_path(train=False)
    test_nocc, test_occ = load_image(test_path)
    model = PDSN_model()
    model.load_weights("model_weight_ck/PDSN_total_train_200")
    nocc , occ = test_nocc.reshape(-1,128,128,1), test_occ.reshape(-1,128,128,1)
    input = tf.concat([nocc, occ], axis=-1)
    _, _, _, _, pred = model(input)
    pred = tf.argmax(pred, axis=-1)
    accuracy = accuracy_score(label , pred)

    return accuracy


if __name__ == "__main__":
    # test(train="total")
    accuracy = test_accuracy()
    print(accuracy)
