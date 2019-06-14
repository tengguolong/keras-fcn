# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:19:04 2019

@author: Teng
"""

from __future__ import print_function
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Reshape
from keras.layers import Activation, Dropout, Conv2DTranspose, Cropping2D
from keras.models import Model
import keras.backend as K

def vgg16(img_input):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x
    
    return [f3, f4, f5]


def crop(x, y):
    h1, w1 = K.int_shape(x)[1:3]
    h2, w2 = K.int_shape(y)[1:3]
    ch = abs(h1 - h2)
    cw = abs(w1 - w2)
    
    if h1 > h2:
        x = Cropping2D(cropping=((ch//2, ch//2),(0,0)), data_format="channels_last")(x)
    elif h1 < h2:
        y = Cropping2D(cropping=((ch//2, ch//2),(0,0)), data_format="channels_last")(y)
    
    if w1 > w2:
        x = Cropping2D(cropping=((0,0),(cw//2, cw//2)), data_format="channels_last")(x)
    elif w1 < w2:
        y = Cropping2D(cropping=((0,0),(cw//2, cw//2)), data_format="channels_last")(y)
    
    return [x, y]            


def fcn8s_head(features, n_classes):
    assert len(features) == 3, 'FCN-8 requires 3 feature maps'
    [f3, f4, f5] = features
    
    x = f5
    x = Conv2D(4096, (7, 7), activation='relu' , padding='same', name='f5_conv1')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(4096, (1, 1), activation='relu' , padding='same', name='f5_conv2')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', name='f5_conv3')(x)
    
    # f5与f4相加
    x = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False, name='deconv1')(x)    
    f4 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', name='f4_conv')(f4)
    x, f4 = crop(x, f4)
    x = Add()([x, f4])
    # 再与f3相加
    x = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False, name='deconv2')(x)
    f3 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', name='f3_conv')(f3)
    x, f3 = crop(x, f3)
    x = Add()([x, f3])
    # 最后反卷积至输入图像尺寸
    x = Conv2DTranspose(n_classes, kernel_size=(16,16), strides=(8,8), use_bias=False, name='deconv3')(x)

    return x


def fcn32s_head(features, n_classes):
    [f3, f4, f5] = features
    
    x = f5
    x = Conv2D(4096, (7, 7), activation='relu' , padding='same', name='f5_conv1')(x)
#    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu' , padding='same', name='f5_conv2')(x)
#    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', name='f5_conv3')(x)
    x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False, name='deconv')(x)
    return x


def fcn8s(height, width, n_classes=21, mode='train'):
    img_input = Input(shape=(height, width, 3))
    output = vgg16(img_input)
    output = fcn8s_head(output, n_classes)
    img_input, output = crop(img_input, output)
    model = Model(img_input, output)
    model.name = 'FCN-8s'
    return model

def fcn32s(height, width, n_classes=21, mode='train'):
    img_input = Input(shape=(height, width, 3))
    output = vgg16(img_input)
    output = fcn32s_head(output, n_classes)
    img_input, output = crop(img_input, output)
    model = Model(img_input, output)
    model.name = 'FCN-32s'
    return model     

if __name__ == '__main__':
    model = fcn8s(512, 512)
    print(model.summary())
