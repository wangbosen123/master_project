import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import options
import matplotlib.pyplot as plt
import cv2
import warp

seed = 456


def load_data_path():
    path = "/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/"
    data = []
    for ID in os.listdir(path):
        for num, filename in enumerate(os.listdir(path + ID)):
            if '-24-' in filename: data.append(path + ID + '/' + filename)
            if '-25-' in filename: data.append(path + ID + '/' + filename)
            if '-26-' in filename: data.append(path + ID + '/' + filename)

    np.random.shuffle(data)
    data = np.array(data)
    return data[0: 2000], data[2000:]

def load_batch_image(data, batch_idx, batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min, range_max))
    train_data = [data[idx] for idx in index]

    images, labels = [], []
    for filename in train_data:
        image = cv2.imread(filename)/255
        images.append(image)
        try:
            labels.append(tf.one_hot(int(filename[64: 66]) - 1, 90))
        except:
            labels.append(tf.one_hot(int(filename[64: 65]) - 1, 90))
    images, labels = np.array(images), np.array(labels)
    return images, labels

def load_mnist(batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.

    # train val split
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=456)


    train_datasets = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).batch(batchSize).shuffle(x_tr.shape[0])
    val_datasets = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchSize).shuffle(x_val.shape[0])
    test_datasets = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize).shuffle(x_test.shape[0])
    return train_datasets, val_datasets, test_datasets


def genPerturbations(opt):
    X = np.tile(opt.canon4pts[:, 0], [opt.batchSize, 1])
    Y = np.tile(opt.canon4pts[:, 1], [opt.batchSize, 1])
    dX = tf.random.normal([opt.batchSize,4], seed=seed)*opt.pertScale + tf.random.normal([opt.batchSize,1], seed=seed)*opt.transScale
    dY = tf.random.normal([opt.batchSize,4], seed=seed)*opt.pertScale + tf.random.normal([opt.batchSize,1], seed=seed)*opt.transScale
    O = np.zeros([opt.batchSize, 4], dtype=np.float32)
    I = np.ones([opt.batchSize, 4], dtype=np.float32)
    # fit warp parameters to generated displacements
    if opt.warpType=="homography":
        A = tf.concat([tf.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
                       tf.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],1)
        b = tf.expand_dims(tf.concat([X+dX,Y+dY],1),-1)
        pPert = tf.compat.v1.matrix_solve(A,b)[:,:,0]
        pPert -= tf.cast([[1,0,0,0,1,0,0,0]], tf.float32)
    else:
        if opt.warpType=="translation":
            J = np.concatenate([np.stack([I,O],axis=-1),
                                np.stack([O,I],axis=-1)],axis=1)
        if opt.warpType=="similarity":
            J = np.concatenate([np.stack([X,Y,I,O],axis=-1),
                                np.stack([-Y,X,O,I],axis=-1)],axis=1)
        if opt.warpType=="affine":
            J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
                                np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
        dXY = tf.expand_dims(tf.concat([dX,dY],1),-1)

        pPert = tf.compat.v1.matrix_solve_ls(J,dXY)[:,:,0]
    return pPert



if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


    #test the coding
    def transformImage_gray(opt, image, pMtrx):
        refMtrx = tf.tile(tf.expand_dims(opt.refMtrx, axis=0), [opt.batchSize, 1, 1])
        transMtrx = tf.matmul(refMtrx, pMtrx)

        # warp the canonical coordinates
        X, Y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 75))
        X, Y = X.flatten(), Y.flatten()
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
        XYhom = np.tile(XYhom, [opt.batchSize, 1, 1]).astype(np.float32)
        XYwarpHom = tf.matmul(transMtrx, XYhom)
        XwarpHom, YwarpHom, ZwarpHom = tf.unstack(XYwarpHom, axis=1)
        Xwarp, Ywarp = tf.cast(tf.reshape(XwarpHom, [opt.batchSize, 75, 50]), tf.int32), tf.cast(tf.reshape(YwarpHom, [opt.batchSize, 75, 50]), tf.int32)


        idxOutside = tf.fill([opt.batchSize, 75, 50], opt.batchSize * 192 * 256)
        imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize, 1, 1]), [1, 75, 50])

        imageVec = tf.reshape(image, [-1, 1])
        imageVecOut = tf.concat([tf.cast(imageVec, dtype='float32'), tf.zeros([1, 1])], axis=0)
        affine_idx = (imageIdx * 192 + Ywarp) * 256 + Xwarp

        def insideImage(Xint, Yint):
            return (Xint >= 0) & (Xint < 192) & (Yint >= 0) & (Yint < 256)

        affine_idx = tf.where(insideImage(Xwarp, Ywarp), affine_idx, idxOutside)
        affine_image = tf.cast(tf.gather(imageVecOut, affine_idx), tf.float32)
        affine_image = tf.image.resize(affine_image, [64, 64])


        print(imageVecOut.shape, affine_idx.shape, affine_image.shape)
        print(affine_image.shape)
        plt.imshow(tf.reshape(affine_image[0], [64, 64]), cmap='gray')
        plt.show()
        plt.imshow(tf.reshape(affine_image[1], [64, 64]), cmap='gray')
        plt.show()
        plt.imshow(tf.reshape(affine_image[2], [64, 64]), cmap='gray')
        plt.show()
        return affine_image


    def transformImage(opt, image, pMtrx):
        refMtrx = tf.tile(tf.expand_dims(opt.refMtrx, axis=0), [opt.batchSize, 1, 1])
        transMtrx = tf.matmul(refMtrx, pMtrx)

        # warp the canonical coordinates
        X, Y = np.meshgrid(np.linspace(-1, 1, opt.W), np.linspace(-1, 1, opt.H))
        X, Y = X.flatten(), Y.flatten()
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
        XYhom = np.tile(XYhom, [opt.batchSize, 1, 1]).astype(np.float32)
        XYwarpHom = tf.matmul(transMtrx, XYhom)
        XwarpHom, YwarpHom, ZwarpHom = tf.unstack(XYwarpHom, axis=1)
        Xwarp = tf.reshape(XwarpHom / (ZwarpHom + 1e-8), [opt.batchSize, opt.H, opt.W])
        Ywarp = tf.reshape(YwarpHom / (ZwarpHom + 1e-8), [opt.batchSize, opt.H, opt.W])

        # get the integer sampling coordinates
        Xfloor, Xceil = tf.floor(Xwarp), tf.math.ceil(Xwarp)
        Yfloor, Yceil = tf.floor(Ywarp), tf.math.ceil(Ywarp)
        XfloorInt, XceilInt = tf.cast(Xfloor, tf.int32), tf.cast(Xceil, tf.int32)
        YfloorInt, YceilInt = tf.cast(Yfloor, tf.int32), tf.cast(Yceil, tf.int32)
        print(XfloorInt)
        print(YfloorInt)

        imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize, 1, 1]), [1, opt.H, opt.W])
        # creat image vector
        imageVec = tf.reshape(image, [-1, int(image.shape[-1])])
        imageVecOut = tf.concat([tf.cast(imageVec, dtype='float32'), tf.zeros([1, int(image.shape[-1])])], axis=0)
        idxUL = (imageIdx * 192 + YfloorInt) * 256 + XfloorInt
        idxUR = (imageIdx * 192 + YfloorInt) * 256 + XceilInt
        idxBL = (imageIdx * 192 + YceilInt) * 256 + XfloorInt
        idxBR = (imageIdx * 192 + YceilInt) * 256 + XceilInt
        # idxOutside shape :(batchsize, 28, 28) >> all element is batchsize*28*28.
        idxOutside = tf.fill([opt.batchSize, opt.H, opt.W], opt.batchSize * 256 * 192)

        def insideImage(Xint, Yint):
            # return (Xint >= 0) & (Xint < opt.W) & (Yint >= 0) & (Yint < opt.H)
            return (Xint >= 0) & (Xint < 192) & (Yint >= 0) & (Yint < 256)

        idxUL = tf.where(insideImage(XfloorInt, YfloorInt), idxUL, idxOutside)
        idxUR = tf.where(insideImage(XceilInt, YfloorInt), idxUR, idxOutside)
        idxBL = tf.where(insideImage(XfloorInt, YceilInt), idxBL, idxOutside)
        idxBR = tf.where(insideImage(XceilInt, YceilInt), idxBR, idxOutside)

        # bilinear interpolation
        Xratio = tf.reshape(Xwarp - Xfloor, [opt.batchSize, opt.H, opt.W, 1])
        Yratio = tf.reshape(Ywarp - Yfloor, [opt.batchSize, opt.H, opt.W, 1])
        imageUL = tf.cast(tf.gather(imageVecOut, idxUL), tf.float32) * (1 - Xratio) * (1 - Yratio)
        imageUR = tf.cast(tf.gather(imageVecOut, idxUR), tf.float32) * (Xratio) * (1 - Yratio)
        imageBL = tf.cast(tf.gather(imageVecOut, idxBL), tf.float32) * (1 - Xratio) * (Yratio)
        imageBR = tf.cast(tf.gather(imageVecOut, idxBR), tf.float32) * (Xratio) * (Yratio)
        imageWarp = imageUL + imageUR + imageBL + imageBR
        # plt.imshow(tf.reshape(imageWarp, [75, 50, 3]), cmap='gray')
        # plt.show()
        print(imageWarp.shape)
        image = tf.image.rgb_to_grayscale(imageWarp)

        plt.imshow(tf.reshape(image[1], [75, 50]), cmap='gray')
        plt.show()
        return imageWarp, XfloorInt, YfloorInt


    # "/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/ID1/m-001-14-0.bmp"
    opt = options.set(training=True)
    opt.batchSize = 3
    init_p = tf.constant([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype='float32')
    # init_p = tf.constant([[0, 0, 0, 0, 0, 0]], dtype='float32')
    pmatrix = warp.vec2mtrx(opt, init_p)

    image1 = cv2.imread("/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/ID1/m-001-14-0.bmp")
    image2 = cv2.imread("/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/ID1/m-001-14-1.bmp")
    image3 = cv2.imread("/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/ID1/m-001-18-0.bmp")
    print(image1.shape)
    image = [image1, image2, image3]
    image = np.array(image)
    # plt.imshow(image, cmap='gray')
    # plt.show()
    transformImage(opt, image, pmatrix)

    # image = cv2.imread("/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/ID1/m-001-14-0.bmp")
    # imagewarp, x, y = warp.transformImage(opt, image, pmatrix)
    # imagewarp = tf.image.rgb_to_grayscale(imagewarp)
    # plt.imshow(tf.reshape(imagewarp, [192, 256]), cmap='gray')
    # plt.show()

    # pad_width = 60
    # X_coord = x + pad_width
    # Y_coord = y + pad_width
    # X_coord = tf.reshape(X_coord, [192 * 256])
    # Y_coord = tf.reshape(Y_coord, [192 * 256])

    # x1, x2, x3, x4 = X_coord[0], X_coord[127], X_coord[128 * 128 - 1 - 127], X_coord[128 * 128 - 1]
    # y1, y2, y3, y4 = Y_coord[0], Y_coord[127], Y_coord[128 * 128 - 1 - 127], Y_coord[128 * 128 - 1]

    # ori = Ori_image[i].numpy().reshape((128, 128)) * 255.
    # print(ori)
    # Ori_img = tf.image.rgb_to_grayscale(image)
    # ori = tf.reshape(Ori_img, [192, 256])
    # padding = np.pad(ori, pad_width, 'constant', constant_values=255)
    # img = tf.reshape(tf.cast(padding, tf.uint8), [192 + pad_width * 2, 256 + pad_width * 2, 1])
    #
    # plt.figure()
    # plt.imshow(img, cmap="gray")
    # plt.plot([x1, x2], [y1, y2], color='red')
    # plt.plot([x2, x4], [y2, y4], color='red')
    # plt.plot([x3, x4], [y3, y4], color='red')
    # plt.plot([x3, x1], [y3, y1], color='red')
    # plt.show()
    # plt.savefig('crop')
    # plt.close('all')


    #face detection.
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # image = cv2.imread('/home/bosen/gradation_thesis/AR_aligment/AR_original_data_clf/ID1/m-001-14-0.bmp')
    # image = cv2.GaussianBlur(image, (5, 5), sigmaX=1, sigmaY=1)
    # image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
    # faces = face_cascade.detectMultiScale(image, 1.1, 4)
    # # Draw the rectangle around each face
    # for (x, y, w, h) in faces:
    #     image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #     # image = image[x:x+w, y:y+h]
    # plt.axis('off')
    # plt.imshow(image)
    # plt.savefig('face_detection',)
    # plt.close()

    # low_img = cv2.resize(img, (8, 8), cv2.INTER_CUBIC)
    # low_ig = cv2.resize(img, (128, 128), cv2.INTER_CUBIC)

    #test simply code
    # O = np.zeros((128))
    # I = np.ones((128))
    # a = np.array([[1, 1], [1, 1], [2, 2]])
    # mask_for_avg = np.concatenate((np.stack([O for i in range(64)]), np.stack([I for i in range(64)])), axis=0)
    # mask_for_affine = np.concatenate((np.stack([I for i in range(64)]), np.stack([O for i in range(64)])), axis=0)
    # print(tf.expand_dims(O, axis=0))
    # print(np.tile(tf.expand_dims(O, axis=0), [30, 1, 1]).astype(np.float32).shape)
    # expand = tf.tile(tf.expand_dims(O, axis=0), [30, 1, 1])
    # print(expand.shape)


