
from keras.layers.convolutional import Convolution2D, Deconvolution2D, AtrousConvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Lambda
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras.applications import vgg16
import keras.backend as K
import numpy as np
import h5py


def load(db_name):
    with h5py.File(db_name, 'r') as hf:
        sketch = np.array(hf['lfw_sketch_data']).astype(np.float32)
        sketch = sketch.transpose((0, 2, 3, 1))
        color = np.array(hf['lfw_64_data']).astype(np.float32) / 255.
        color = color.transpose((0, 2, 3, 1))
        weights = np.array(hf['vgg_16_weight'])
        print 'sketch data has shape:', sketch.shape
        print 'color data has shape:', color.shape
        print 'vgg_16 weights data has shape', weights.shape
        return sketch, color, weights


def residual_block(x, block_idx, nb_filter, bn=True, weight_decay=0, k_size=3):

    # 1st conv
    name = "block%s_conv2D%s" % (block_idx, "a")
    W_reg = l2(weight_decay)
    r = Convolution2D(nb_filter, k_size, k_size, border_mode="same", W_regularizer=W_reg, name=name)(x)
    if bn:
        r = BatchNormalization(mode=2, axis=1, name="block%s_bn%s" % (block_idx, "a"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "a"))(r)

    # 2nd conv
    name = "block%s_conv2D%s" % (block_idx, "b")
    W_reg = l2(weight_decay)
    r = Convolution2D(nb_filter, k_size, k_size, border_mode="same", W_regularizer=W_reg, name=name)(r)
    if bn:
        r = BatchNormalization(mode=2, axis=1, name="block%s_bn%s" % (block_idx, "b"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "b"))(r)

    # Merge residual and identity
    x = merge([x, r], mode='sum', concat_axis=1, name="block%s_merge" % block_idx)

    return x


def convolutional_block(x, block_idx, nb_filter, k_size=3, subsample=(1, 1)):
    name = "block%s_conv2D" % block_idx
    x = Convolution2D(nb_filter, k_size, k_size, name=name, border_mode="same", subsample=subsample)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("relu")(x)
    return x


def deconvolutional_block(x, block_idx, nb_filter, output_shape, k_size=3, subsample=(2, 2)):
    name = "block%s_deconv2D" % block_idx
    x = Deconvolution2D(nb_filter, k_size, k_size, output_shape=output_shape, name=name, border_mode='same', subsample=subsample)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("relu")(x)
    return x


def edge2color(img_dim, batch_size):
    x_input = Input(shape=img_dim, name='input')
    h1 = convolutional_block(x_input, 1, 32, k_size=9, subsample=(1, 1))
    h2 = convolutional_block(h1, 2, 64, k_size=3, subsample=(2, 2))
    h3 = convolutional_block(h2, 3, 128, k_size=3, subsample=(2, 2))
    h4 = residual_block(h3, 4, 128, k_size=3)
    h5 = residual_block(h4, 5, 128, k_size=3)
    h6 = residual_block(h5, 6, 128, k_size=3)
    h7 = residual_block(h6, 7, 128, k_size=3)
    h8 = residual_block(h7, 8, 128, k_size=3)
    h9 = deconvolutional_block(h8, 9, 64, k_size=3, output_shape=(batch_size, 32,32,64), subsample=(2, 2))
    h10 = deconvolutional_block(h9, 10, 32, k_size=3, output_shape=(batch_size, 64,64,32), subsample=(2, 2))
    h11 = convolutional_block(h10, 11, 3, k_size=9, subsample=(1, 1))
    # with K.get_session():
    vgg_16 = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None)
    f = vgg_16(h11)
    model = Model(input=x_input, output=[h11, f], name='edge2color')
    return model, model.name










