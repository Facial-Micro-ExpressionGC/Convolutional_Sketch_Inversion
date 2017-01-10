from keras.optimizers import Adam
from keras.utils import generic_utils
from ops import *
import keras.backend as K
import sys
# Utils
sys.path.append("utils/")
from simple_utils import plot_batch

import os
import time


def feature_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def pixel_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def train(batch_size, n_batch_per_epoch, nb_epoch, sketch, color, weights, tag, save_weight=1):
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model, model_name = edge2color([64,64,1], batch_size=batch_size)

    model.compile(loss=[pixel_loss, feature_loss], loss_weights=[5, 1], optimizer=opt)
    model.summary()
    from keras.utils.visualize_util import plot
    plot(model, to_file='figures/edge2color.png', show_shapes=True, show_layer_names=True)

    global_counter = 1
    for epoch in range(nb_epoch):
        batch_counter = 1
        start = time.time()
        batch_idxs = sketch.shape[0] // batch_size
        if n_batch_per_epoch >= batch_idxs or n_batch_per_epoch == 0:
            n_batch_per_epoch = batch_idxs
        progbar = generic_utils.Progbar(n_batch_per_epoch * batch_size)
        for idx in range(batch_idxs):
            batch_sk = sketch[idx * batch_size: (idx + 1) * batch_size]
            batch_co = color[idx * batch_size: (idx + 1) * batch_size]
            batch_weights = weights[idx * batch_size: (idx + 1) * batch_size]
            train_loss = model.train_on_batch([batch_sk], [batch_co, batch_weights])
            batch_counter += 1
            progbar.add(batch_size, values=[('pixel_loss', train_loss[0]), ('feature_loss', train_loss[1])])
            # if batch_counter >= n_batch_per_epoch:
            if global_counter % 50 == 1:
                plot_batch(model, batch_size, batch_sk, batch_co, epoch, idx, tag)
            global_counter += 1

            if batch_counter >= n_batch_per_epoch:
                break
        print ""
        print 'Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start)

        if save_weight:
            # save weights every epoch
            weights_path = '%s/%s_weights_epoch_%s.h5' % (model_name, model_name, epoch)
            if not os.path.exists('%s' % model_name):
                os.mkdir('%s' % model_name)
            model.save_weights(weights_path, overwrite=True)


if __name__ == '__main__':
    sketch, color, weights = load(os.path.expanduser('~/Desktop/hdf5/clear/database_train.h5'))
    train(batch_size=64, n_batch_per_epoch=1, nb_epoch=1000, sketch=sketch, color=color,
          weights=weights, tag='5-1-1batch', save_weight=0)
