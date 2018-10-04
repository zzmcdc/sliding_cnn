#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:31:41 2018

@author: zhao
"""

from __future__ import print_function

import logging
import os
import argparse

import mxnet as mx
import numpy as np
from PIL import Image
from functools import reduce
from cnn_sliding_2 import get_symbol
from cnn_sliding_dataset import OCRIter
from config import train_param
import editdistance
from ctc_metrics import CtcMetrics

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_config', required=True,type=str, help='a yaml file for trainning crnn')

command_arg = parser.parse_args()
config = train_param()
config.load_parm(command_arg.train_config)
param = config.get_parm()


with open(param['charset']) as to_read: 
    classes = list(to_read.read().strip())
num_classes = len(classes) + 1


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break

        ret.append(l[i])
    return ret




def get_string(label_list):
    ret = []
    label_list2 = [0] + list(label_list)
    for i in range(len(label_list)):
        c1 = label_list2[i]
        c2 = label_list2[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
        # change to ascii
    s = ''
    for l in ret:
        if l > 0 and l < (len(classes) + 1):
            c = classes[l - 1]
        else:
            c = ''
        s += c
    return s

def num2str(label_list):
    label_list = remove_blank(label_list)
    s=''
    for item in label_list:
        s += classes[int(item)-1]
    return s


def Accuracy(label, pred):
    hit = 0.
    total = 0.
    for i in range(param['batch_size']):
        l = remove_blank(label[i])
        p = []
        for k in range(param['seq_len']):
            p.append(np.argmax(pred[k * param['batch_size'] + i]))
        p = ctc_label(p)
        #if len(p) == len(l):
        #    for k in range(len(p)):
        #        if p[k] == int(l[k]):
        #            hit += 1.0

        #else:
        #    for j in range(min(len(p),len(l))):
        #        if p[j] == l[j]:
        #            hit += 1.0
        for j in range(min(len(p),len(l))):
            if p[j] == l[j]:
                hit += 1.0
        total += len(l)
    return hit / total


def edit_distance(label, pred):
    new_prob = np.reshape(pred, (param['seq_len'], param['batch_size'], -1))
    new_prob = np.swapaxes(new_prob, 0, 1)
    label_list = np.argmax(new_prob, axis=2)
    result = []
    target = []
    for i in range(param['batch_size']):
        result.append(get_string(label_list[i]))
        target.append(num2str(label[i]))
    score = 0
    for s,d in zip(target,result):
        score += editdistance.eval(s,d)
    return (score)/param['batch_size']


def norm_stat(d):
#    return d
    return mx.nd.norm(d)/(np.sqrt(d.size) +1)

def mean_stat(x):
    return mx.nd.mean(x)

def mean_abs(x):
    return mx.ndarray.divide(mx.ndarray.sum(mx.ndarray.abs(x)), reduce(lambda x, y: x * y, x.shape));

class single_loss_metric(mx.metric.EvalMetric):
    def __init__(self,name='loss'):
        super(single_loss_metric, self).__init__(name)
    def update(self,labels,preds):
        for pred in preds[1]:
            self.sum_metric+=pred.asnumpy().sum()
            self.num_inst+=pred.shape[0]

if __name__ == '__main__':

    opt = config.get_parm()
    
    print(opt)
    
    model_name = opt['name']
    log_file_name = model_name+'.log'
    log_file = open(log_file_name, 'w')
    log_file.close()
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_name)
    logger.addHandler(fh)
    model_dir_path = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    prefix = os.path.join(os.getcwd(), 'model', model_name)


    num_label = opt['num_label']
    data_names = ['data',]
    data_shape = (opt['imgW'], opt['imgH'])


    gpu_list=[0,1,2,3,4,5]
    data_train = OCRIter(batch_size = opt['batch_size']*len(gpu_list), classess = classes, dataset_lst = opt['train_lst'])
    data_val   = OCRIter(batch_size = opt['batch_size']*len(gpu_list), classess = classes, dataset_lst = opt['val_lst'])

    ctx = [mx.gpu(i) for i in gpu_list] #if opt.gpu else mx.cpu(0)
 
    label_names = ['label', ]
    if opt['from_epoch'] ==0:
        opt['from_epoch'] = None
        
    if opt['from_epoch'] is None:
        symbol = get_symbol(seq_len=opt['seq_len'],num_label=num_label,num_class=num_classes,true_batch=opt['batch_size'])
        model = mx.module.Module(
            symbol=symbol,
            data_names=data_names,
            label_names=label_names,
            context=ctx
        )
    else:
        model = mx.module.Module.load(
            prefix,
            opt['from_epoch'],
            data_names=data_names,
            label_names=label_names,
            context=ctx,
        )

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    logger.info('begin fit')
    mon = mx.mon.Monitor(10,norm_stat,pattern=r'.*?warpctc0_output.*?')
    metrics = CtcMetrics(opt['seq_len'])
    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(single_loss_metric())
    #eval_metric.add(mx.metric.np(Accuracy))
    eval_metric.add(mx.metric.np(edit_distance, allow_extra_outputs=True))
    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=eval_metric,
        batch_end_callback=mx.callback.Speedometer(opt['batch_size']*len(gpu_list), 100),
        epoch_end_callback=mx.callback.do_checkpoint(prefix, 1),
        #monitor=mon,        
        optimizer='adam',
        optimizer_params={'learning_rate': opt['learning_rate'],'clip_gradient':5},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.4),
        num_epoch=opt['num_epoch'],
        begin_epoch=opt['from_epoch'] if opt['from_epoch'] else 0
    )
    model.save_params(model_name)
