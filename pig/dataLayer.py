# -*- coding:utf-8 -*-

import sys
import caffe
import numpy as np
import os
import cv2

from caffe import layers as L, params as P
from caffe.coord_map import crop
from tools import SimpleTransformer
from random import shuffle
from PIL import Image
from threading import Thread

class MultiImageDataLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        
        self.top_names = ['ori_data', 'pig_data', 'part1_data', 'part2_data', 'part3_data', 'part4_data', 'label']

        # 读prototxt中参数
        params = eval(self.param_str)
        
        # Check the parameters for validity
        check_params(params)

        self.batch_size = int(params['batch_size'])
        
        # Create a batch loader to load images
        self.batch_loader = BatchLoader(params, None)

        # seven tops
        if len(top) != 7:
            raise Exception('Need to define seven tops: ori_data, pig_data, part[1-4]_data, label')

        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception('Do not define a bottom.')
        
        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['ori_shape'][0], params['ori_shape'][1])
        top[1].reshape(self.batch_size, 3, params['pig_shape'][0], params['pig_shape'][1])

        for i in range(2, 6):
            top[i].reshape(self.batch_size, 3, params['part_shape'][0], params['part_shape'][1])

        top[6].reshape(self.batch_size, 1)

        print_info("MultiImageDataLayerSync", params)
    
    def forward(self, bottom, top):
        """
        Load data
        """

        for itt in range(self.batch_size):
            ims , label = self.batch_loader.load_next_image()
            
            #  Add directly to the caffe data layer
            for i in range(6):
                top[i].data[itt, ...] = ims[i]
            
            top[6].data[itt, ...] = label


    def reshape(self, bottom, top):
        """
        Reshape the data
        """
        pass
    
    def backward(self, top, propagate_down, bottom):
        """
        Back propagate
        """
        pass


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    """
    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        # get list of image indexs
        self.indexlist = [line.rstrip('\n') for line in open(params['data']).readlines()]

        self._cur = 0 # current image
        # this class does some simple data-manipulation
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load images
        index_line = self.indexlist[self._cur] # Get the image index
        
        indexs = index_line.split("\t")

        images = []

        for index in indexs[:-1]:
            im = np.asarray(Image.open(index))

            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]
            
            images.append(self.transformer.preprocess(im))

        # Load and prepare ground truth
        #label = np.zeros(self.gt_classes).astype(np.float32)
        label = int(indexs[-1])-1

        self._cur += 1
        return images, label
            
def check_params(params):
    """
    A utilty function to check the parameters for the data layers
    """
    assert 'data' in params.keys(), 'Params must include data indexs'

    required = ['data','batch_size',  'ori_shape', 'pig_shape', 'part_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)

def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, bs: {}, gt_classes:{}".format(
        name,
        params['data'],
        params['batch_size'],
        #params['gt_classes'],
        params['ori_shape'],
        params['pig_shape'],
        params['part_shape'])
