#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:18:31 2018

@author: zhao
"""

import mxnet as mx
import cv2

def conv_bn_re(data, num_filter,
               kernel=(3,3), stride=(1,1), pad=(1,1),name_prefix=''):
    net = mx.sym.Convolution(data=data, kernel=kernel,
                             stride=stride, num_filter=num_filter,
                             name=name_prefix+'_conv',pad=pad)
#    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name=name_prefix + '_bn')
#    net = mx.sym.Activation(data=net, act_type='relu', name=name_prefix+'_relu')
    return net

 
def get_symbol(seq_len, num_label,num_class,true_batch=4):
    data = mx.sym.Variable('data')
#    data = mx.sym.Reshape(data=data,shape=(-1,1,32,32))
    
    net = conv_bn_re(data=data, num_filter=50, name_prefix='layer_1')
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_1' + '_bn')
    net = mx.sym.Activation(data=net, act_type='relu', name='layer_1'+'_relu')
    
    net = conv_bn_re(data=net, num_filter=100, name_prefix='layer_2')
    net = mx.sym.Activation(data=net, act_type='relu', name='layer_2'+'_relu')
    net = mx.sym.Dropout(data=net, p=0.1)
    
    net = conv_bn_re(data=net, num_filter=100, name_prefix='layer_3')
    net = mx.sym.Activation(data=net, act_type='relu', name='layer_3'+'_relu')
    net = mx.sym.Dropout(data=net, p=0.1)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_3' + '_bn')
    
    
    net = mx.sym.Pooling(data=net, kernel=(2,2),stride=(2,2), pool_type='max', name='pool1')
    
    net = conv_bn_re(data=net, num_filter=150,name_prefix='layer_4')
    net = mx.sym.Dropout(data=net,p=0.2)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_4' + '_bn')
    
    net = conv_bn_re(data=net, num_filter=200, name_prefix='layer_5')
    net = mx.sym.Dropout(data=net, p=0.2)
    
    net = conv_bn_re(data=net, num_filter=200, name_prefix='layer_6')
    net = mx.sym.Dropout(data=net, p=0.2)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_6' + '_bn')
    
    net = mx.sym.Pooling(data=net, kernel=(2,2),stride=(2,2), pool_type='max', name='pool2')
    
    net = conv_bn_re(data=net,num_filter=250, name_prefix='layer_7')
    net = mx.sym.Dropout(data=net, p=0.3)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_7' + '_bn')
    
    net = conv_bn_re(data=net, num_filter=300, name_prefix='layer_8')
    net = mx.sym.Dropout(data=net, p=0.3)
    
    net = conv_bn_re(data=net, num_filter=300, name_prefix='layer_9')
    net = mx.sym.Dropout(data=net, p=0.3)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_9' + '_bn')
    
    net = mx.sym.Pooling(data=net,kernel=(2,2),stride=(2,2),
                         pool_type='max', name='pool3')
    
    net = conv_bn_re(data=net, num_filter=350, name_prefix='layer_10')
    net = mx.sym.Dropout(data=net, p=0.3)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_10' + '_bn')
    
    net = conv_bn_re(data=net, num_filter=400, name_prefix='layer_11')
    net = mx.sym.Dropout(data=net,p=0.4)
    
    net = conv_bn_re(data=net, num_filter=400, name_prefix='layer_12')
    net = mx.sym.Dropout(data=net, p=0.4)
    net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name='layer_12' + '_bn')
    
    net = mx.sym.Pooling(data=net, kernel=(2,2), stride=(2,2),
                         pool_type='max', name='pool4')
    
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=4096,name='fc1')
    net = mx.sym.Dropout(data=net,p=0.5)
    net = mx.sym.FullyConnected(data=net, num_hidden=num_class)
    net = mx.sym.Activation(data=net, act_type='relu', name='fc'+'_relu')
 
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(data=label, shape=(-1,)) 
    label = mx.sym.Cast(data=label,dtype='int32')
    label = mx.sym.Reshape(data=label, shape=(-1))
    label = mx.sym.slice(data=label, begin=(0),end=(num_label*true_batch))
    loss = mx.sym.WarpCTC(data=net, label=label, label_length=num_label, input_length=seq_len)
    
    return loss

if __name__=='__main__':
    sym = sym = get_symbol(16,40, 5285)
    arg_name = sym.list_arguments()
    out_name = sym.get_internals().list_outputs()
    mx.viz.print_summary(symbol=sym,shape={'data':(1,60,32,512)})
    arg_shape, out_shape, _ = sym.get_internals().infer_shape(data=(1,60,32,256))
    for name, shape in zip(out_name,out_shape):
        print(name,':',shape)

