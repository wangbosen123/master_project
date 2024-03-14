from prepare_data import *
import scipy.linalg
import tensorflow as tf
import numpy as np
from build_model import *
import matplotlib.cm as cm
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def gradcam_heatmap_mutiple(img_array, model, last_conv_layer_name, network_layer_name, label, corresponding_label=True):
	label = np.argmax(label, axis=-1)
	last_conv_layer = model.get_layer(last_conv_layer_name)
	last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

	network_input = Input(shape=last_conv_layer.output.shape[1:])
	x = network_input
	for layer_name in network_layer_name:
		x = model.get_layer(layer_name)(x)
	network_model = Model(network_input, x)

	with tf.GradientTape() as tape:
		last_conv_layer_output = last_conv_layer_model(img_array)
		tape.watch(last_conv_layer_output)
		preds = network_model(last_conv_layer_output)
		if corresponding_label:
			pred_index = tf.constant(label, dtype=tf.int64)
		else:
			pred_index = np.argmax(preds, axis=-1)
		class_channel = tf.gather(preds, pred_index, axis=-1, batch_dims=1)
	grads = tape.gradient(class_channel, last_conv_layer_output)
	pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

	pooled_gradsa = tf.tile(tf.reshape(pooled_grads, [pooled_grads.shape[0], 1, 1, pooled_grads.shape[1]]), [1, last_conv_layer_output.shape[1], last_conv_layer_output.shape[2], 1])
	heatmap = last_conv_layer_output * pooled_gradsa
	heatmap = tf.reduce_sum(heatmap, axis=-1)
	heatmap_min, heatmap_max = [], []
	for num in range(heatmap.shape[0]):
		heatmap_min.append(np.min(heatmap[num]))
		heatmap_max.append(np.max(heatmap[num]))
	heatmap_min, heatmap_max = tf.cast(heatmap_min, dtype=tf.float32), tf.cast(heatmap_max, dtype=tf.float32)
	heatmap_min = tf.tile(tf.reshape(heatmap_min, [heatmap_min.shape[0], 1, 1]), [1, heatmap.shape[1], heatmap.shape[2]])
	heatmap_max = tf.tile(tf.reshape(heatmap_max, [heatmap_max.shape[0], 1, 1]), [1, heatmap.shape[1], heatmap.shape[2]])
	heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-20)

	heatmap_gray = tf.cast(heatmap, dtype=tf.float32)
	heatmap = np.uint8(255 * heatmap)

	cmap = cm.get_cmap("jet")
	cmap_colors = cmap(np.arange(256))[:, :3]
	cmap_heatmap = cmap_colors[heatmap]
	cmap_heatmap = tf.image.resize(cmap_heatmap, [64, 64], method='bicubic')
	heatmap_gray = tf.image.resize(tf.reshape(heatmap_gray, [-1, 32, 32, 1]), [64, 64], method='bicubic')
	return cmap_heatmap, heatmap_gray

def fit(Xsrc, Xdst):
	ptsN = len(Xsrc)
	X, Y, U, V, O, I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
	A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1), np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
	b = np.concatenate((U,V),axis=0)
	p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
	pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
	return pMtrx

def compose(batch_size, p, dp):
	pMtrx = vec2mtrx(batch_size,p)
	dpMtrx = vec2mtrx(batch_size,dp)
	pMtrxNew = tf.matmul(dpMtrx,pMtrx)
	pMtrxNew /= pMtrxNew[:,2:3,2:3]
	pNew = mtrx2vec(pMtrxNew)
	return pNew

def vec2mtrx(batch_size, p):
	O = tf.zeros([batch_size])
	I = tf.ones([batch_size])
	p1,p2,p3,p4,p5,p6 = tf.unstack(p,axis=1)
	pMtrx = tf.transpose(tf.stack([[I+p1,p2,p3],[p4,I+p5,p6],[O,O,I]]), perm=[2,0,1])
	return pMtrx

def mtrx2vec(pMtrx):
	[row0,row1,row2] = tf.unstack(pMtrx,axis=1)
	[e00,e01,e02] = tf.unstack(row0,axis=1)
	[e10,e11,e12] = tf.unstack(row1,axis=1)
	[e20,e21,e22] = tf.unstack(row2,axis=1)
	p = tf.stack([e00-1,e01,e02,e10,e11-1,e12],axis=1)
	return p

def transformImage(image, refMtrx, pMtrx, batch_size, w, h):
	# refMtrx = tf.tile(tf.expand_dims(refMtrx, axis=0), [batch_size, 1, 1])
	transMtrx = tf.matmul(refMtrx, pMtrx)

	X,Y = np.meshgrid(np.linspace(0, w, h), np.linspace(0, w, h))
	X,Y = X.flatten(), Y.flatten()
	XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
	XYhom = np.tile(XYhom, [batch_size, 1, 1]).astype(np.float32)
	XYwarpHom = tf.matmul(transMtrx, XYhom)
	XwarpHom, YwarpHom, ZwarpHom = tf.unstack(XYwarpHom, axis=1)
	Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8), [batch_size, w, h])
	Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8), [batch_size, w, h])

	Xfloor, Xceil = tf.floor(Xwarp), tf.math.ceil(Xwarp)
	Yfloor, Yceil = tf.floor(Ywarp), tf.math.ceil(Ywarp)
	XfloorInt, XceilInt = tf.cast(Xfloor, tf.int32), tf.cast(Xceil, tf.int32)
	YfloorInt, YceilInt = tf.cast(Yfloor, tf.int32), tf.cast(Yceil, tf.int32)

	imageIdx = np.tile(np.arange(batch_size).reshape([batch_size, 1, 1]), [1, w, h])
	imageVec = tf.reshape(image, [-1, int(image.shape[-1])])
	imageVecOut = tf.concat([tf.cast(imageVec, dtype='float32'), tf.ones([1, int(image.shape[-1])])], axis=0)
	idxUL = (imageIdx * 192 + YfloorInt) * 256 + XfloorInt
	idxUR = (imageIdx * 192 + YfloorInt) * 256 + XceilInt
	idxBL = (imageIdx * 192 + YceilInt) * 256 + XfloorInt
	idxBR = (imageIdx * 192 + YceilInt) * 256 + XceilInt
	idxOutside = tf.fill([batch_size, w, h], batch_size * 192 * 256)

	def insideImage(Xint, Yint):
		return (Xint >= 0) & (Xint < 192) & (Yint >= 0) & (Yint < 256)
	idxUL = tf.where(insideImage(XfloorInt, YfloorInt), idxUL, idxOutside)
	idxUR = tf.where(insideImage(XceilInt, YfloorInt), idxUR, idxOutside)
	idxBL = tf.where(insideImage(XfloorInt, YceilInt), idxBL, idxOutside)
	idxBR = tf.where(insideImage(XceilInt, YceilInt), idxBR, idxOutside)

	Xratio = tf.reshape(Xwarp - Xfloor, [batch_size, w, h, 1])
	Yratio = tf.reshape(Ywarp - Yfloor, [batch_size, w, h, 1])
	imageUL = tf.cast(tf.gather(imageVecOut, idxUL), tf.float32) * (1 - Xratio) * (1 - Yratio)
	imageUR = tf.cast(tf.gather(imageVecOut, idxUR), tf.float32) * (Xratio) * (1 - Yratio)
	imageBL = tf.cast(tf.gather(imageVecOut, idxBL), tf.float32) * (1 - Xratio) * (Yratio)
	imageBR = tf.cast(tf.gather(imageVecOut, idxBR), tf.float32) * (Xratio) * (Yratio)
	imageWarp = imageUL + imageUR + imageBL + imageBR
	imageWarp = tf.image.rgb_to_grayscale(imageWarp)
	return imageWarp


if __name__ == "__main__":
	pass
