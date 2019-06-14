# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:17:06 2019

@author: Teng
"""

from __future__ import print_function
import os
import cv2
from PIL import Image
import pickle
import numpy as np
import argparse
from datetime import datetime as timer
from keras import backend as K

from model import fcn8s
from config import cfg
from vis import make_palette, color_seg, vis_seg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a classifier network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        type=str)
    args = parser.parse_args()
    return args


def pred(model, img_path):
    input_images = [] # Store resized versions of the images here.
    orig_image = cv2.imread(img_path)
    img = cv2.resize(orig_image, (cfg.height, cfg.width)).astype('float32')
    img -= cfg.mean
    input_images.append(img)
    input_images = np.array(input_images)   
    
    y_pred = model.predict(input_images)
    seg = np.argmax(y_pred[0], axis=-1)
    return seg
    

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:\n', args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    K.clear_session() # Clear previous models from memory.
    model = fcn8s(height=cfg.height, width=cfg.width)
    weights_path = os.path.abspath(args.weights)
    model.load_weights(weights_path, by_name=True)
    
    with open('../data/VOC2007/ImageSets/Segmentation/minival.txt') as f:
        ims = [x.strip()+'.jpg' for x in f.readlines()][:10]
        
    t = timer.now()
    segs = []
    for i in range(len(ims)):
        img_path = '../data/VOC2007/JPEGImages/'+ims[i]  
        segs.append(pred(model, img_path))
    delta = timer.now() - t
    print(len(ims), 'images cost: ', delta)
    print('average cost: ', delta/len(ims))
    
    # visualize segmentation in PASCAL VOC colors
    voc_palette = make_palette(21)
    for i in range(len(ims)):
        out_im = Image.fromarray(color_seg(segs[i], voc_palette))
        out_im.save('results/{}_output.png'.format(ims[i][:-4]))
#        masked_im = Image.fromarray(vis_seg(im, out, voc_palette))
#        masked_im.save('demo/visualization.jpg')
    
    


