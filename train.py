# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:25:28 2019

@author: Teng
"""

from __future__ import print_function
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
import os
import argparse

from config import cfg
from model import fcn8s_resnet
from data_generator import DataGenerator


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FCN model')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--init', dest='init_epoch',
                        help='initial epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='train the dataset how many times',
                        default=1, type=int)
    parser.add_argument('--net', dest='base_net',
                        help='base net to extract feature',
                        default='vgg16', type=str)
    parser.add_argument('--no_weights', action='store_true',
                        default=False, dest='no_weights')
    parser.add_argument('--load_model', action='store_true',
                        default=False, dest='load_model')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default='snapshot/vgg16_weights_notop.h5', type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='num images used in one update',
                        default=4, type=int)
    args = parser.parse_args()
    return args


def fcnLoss(y_true, y_pred):
    '''input shape: true(b, h, w)  pred(b, h, w, n_classes)'''    
    # 提取标签不为255的样本
    y_true = K.reshape(y_true, shape=(-1, cfg.height, cfg.width))
    non_negative_inds = tf.where(K.not_equal(y_true, 255)) # (None, 3)
    y_true = tf.gather_nd(y_true, non_negative_inds) # (None,)
    y_pred = tf.gather_nd(y_pred, non_negative_inds) # (None, n_classes)
    
    y_true = K.one_hot(K.cast(y_true, dtype='int32'), cfg.n_classes) # (None, n_classes)
    assert K.int_shape(y_true) == K.int_shape(y_pred), '{}, {}'.format(K.int_shape(y_true), K.int_shape(y_pred))
    return K.mean(K.categorical_crossentropy(y_true, K.softmax(y_pred)))


# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch < 10:
        return 1e-4
    elif epoch < 30:
        return 1e-5
    else:
        return 2e-6


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:\n', args)
    
     # allocate GPU resources
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    sess = tf.Session(config=c)
    K.tensorflow_backend.set_session(sess)
    
    # Build model
    K.clear_session()
    if args.load_model:
        print('Loading model from ', os.path.abspath(args.weights))
        model = load_model(args.weights, custom_objects={'fcnLoss': fcnLoss})
        print(model.summary())
    else:
        model = fcn8s_resnet(height=cfg.height, width=cfg.width)
        print(model.summary())
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(optimizer=sgd, loss=fcnLoss)    
        # Load weights
        if not args.no_weights:
            print('Loading weights from ', os.path.abspath(args.weights))
            model.load_weights(args.weights, by_name=True)
        
    # Define model callbacks.
    model_checkpoint = ModelCheckpoint(filepath='snapshot/fcn8s_resnet_epoch-{epoch:02d}_loss-{loss:.3f}_val_loss-{val_loss:.3f}.h5',
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=False,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=10) 
    csv_logger = CSVLogger(filename='log/fcn8s_resnet_training_log.csv',
                           separator=',',
                           append=True)
    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [model_checkpoint,
                 csv_logger,
                 learning_rate_scheduler,
                 terminate_on_nan]
    
    train_generator = DataGenerator(split='train', batch_size=args.batch_size)
    val_generator = DataGenerator(split='minival', batch_size=1, shuffle=False)
    history = model.fit_generator(generator=train_generator,
                                  epochs=args.epochs,
                                  callbacks=callbacks,
                                  workers=1,
                                  validation_data=val_generator,
                                  initial_epoch=args.init_epoch)
    
        



