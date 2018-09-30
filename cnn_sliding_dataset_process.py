#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:24:50 2018

@author: zhao
"""

import mxnet as mx
import cv2
import numpy as np
import random
import multiprocessing as mp


class SimpleBatch(object):
    def __init__(self, data, data_names, label, label_names):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.pad = 0
        self.index = 0

    @property
    def provide_data(self):
        return [(n,x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n, x in zip(self.label_names, self.label)]


def sliding_generate_batch_layer(inputs,character_width=32,character_step=8):
    # inputs: batches*32*280
    chanel = len(list(range(0,inputs.shape[2] - character_width, character_step)))
    out_put = np.zeros(shape=(inputs.shape[0],chanel,character_width,character_width))
    for b in range(inputs.shape[0]):
        batch_input=inputs[b,:,:].reshape((1,inputs.shape[1],inputs.shape[2]))
        for w in range(0,batch_input.shape[2]-character_width,character_step):
            if w==0:
                output_batch=batch_input[:,:,w:(w+1)*character_width]
            else:
                output_batch=np.concatenate((output_batch,batch_input[:,:,w:w+character_width]),axis=0)
        out_put[b] = output_batch
#    output = np.transpose(output,axes=[0,3,2,1])
#    output = np.reshape(output,(inputs.shape[0],-1,character_width,character_width))
    
    return out_put



class OCRIter(mx.io.DataIter):
    def __init__(self, batch_size, classess,dataset_lst, data_shape=(512,32), character_shape=(32,32), num_label=40, character_step=8):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.character_shape = character_shape
        self.num_label = num_label
        self.classes = classess
        self.dataset_lst = dataset_lst
        self.character_shape = character_shape
        self.character_step = character_step

        self.seq_len = len(range(0, data_shape[0] - character_shape[0], character_step))
        self.batch_len = self.batch_size * self.seq_len

        self.provide_data = [('data', (self.batch_size, 60, character_shape[0], character_shape[1]))]
        self.provide_label = [('label', (self.batch_size, num_label))]

        with open(self.dataset_lst, 'r') as f:
            self.train_set = f.readlines()
        random.shuffle(self.train_set)
        print(len(self.train_set))

    def __iter__(self):
        label = []
        cnt = 0
        random.shuffle(self.train_set)

        img_batch = np.zeros(shape=(self.batch_size, self.data_shape[1], self.data_shape[0]))
        for m_line in self.train_set:
            img_path, img_label = m_line.strip().split(' ')
            img_label = img_label.strip('"')
            plate_str = img_label
            ret = np.zeros(self.num_label, int)
            for number in range(len(plate_str)):
                 ret[number] = self.classes.index(plate_str[number]) + 1
            label.append(ret)
#            print(img_path)
            img = cv2.imread(img_path, 0)
#            h = img_shape[0]
#            w = img_shape[1]
#            new_h = self.data_shape[1]
#            new_w = int(new_h *(w/h))

            img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
#            cv2.imshow('test',img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            img = np.expand_dims(img, axis=2)
#            img_new[0:new_h,0:new_w] = img
            img_batch[cnt] = img
            
            cnt +=1

            if cnt % self.batch_size == 0:
                cnt = 0
                data_all = sliding_generate_batch_layer(img_batch)
                data_all = mx.nd.array(data_all)
                data_names = ['data']
                label_names = ['label']
                label_all = mx.nd.array(label)
                ipdb.set_trace()
                yield SimpleBatch([data_all], data_names, [label_all], label_names)
                continue
    def reset(self):
        random.shuffle(self.train_set)
if __name__=='__main__':
    with open('label.txt','r') as to_read:
        classes = list(to_read.read().strip())
    data_iter = OCRIter(64,classes,'/home/zhao/IIIT5K/train_check.txt')
    it = iter(data_iter)
    data = next(it)


