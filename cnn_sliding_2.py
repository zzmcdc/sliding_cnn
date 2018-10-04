#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:57:13 2018

@author: zhao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:18:31 2018

@author: zhao
"""

import mxnet as mx
import cv2


def conv_bn_re(data, num_filter,dropout=None,bn=False,relu =False,kernel=(3,3), stride=(1,1), pad=(1,1),name_prefix=''):
    net = mx.sym.Convolution(data=data, kernel=kernel,
                             stride=stride, num_filter=num_filter,
                             name=name_prefix+'_conv',pad=pad)
    if dropout is not None:
        net = mx.sym.Dropout(data=net,p=dropout)
    if bn:
        net = mx.sym.BatchNorm(data=net,fix_gamma=False, eps=2e-5, momentum=0.9, name=name_prefix + '_bn')
    if relu:
        net = mx.sym.Activation(data=net, act_type='relu', name=name_prefix+'_relu')
    return net


def get_symbol(seq_len, num_label,num_class,true_batch=4):
    data = mx.sym.Variable('data')
#    data = mx.sym.split(data,num_outputs=60,axis=1)
#    data = mx.sym.concat(*data,dim=0)
    
    net = conv_bn_re(data=data,num_filter = 50,bn=True,relu=True, name_prefix='conv1')
    
    net = conv_bn_re(data=net, num_filter=100,dropout=0.1,relu=True,name_prefix='conv2')
    
    net = conv_bn_re(data=net, num_filter=100,dropout=0.1,bn=True,relu=True, name_prefix='conv3')
    
    net = mx.sym.Pooling(data=net, kernel=(2,2),stride=(2,2), pool_type='max', name='pool3')
    
    
    net = conv_bn_re(data=net,num_filter=150, dropout=0.2,bn=True,relu=True, name_prefix='conv4')
    
    net = conv_bn_re(data=net, num_filter=200,dropout=0.2, name_prefix='conv5')
    
    net = conv_bn_re(data=net, num_filter=200, dropout=0.2, bn=True, relu=True, name_prefix='conv6')
    
    net = mx.sym.Pooling(data=net, kernel=(2,2),stride=(2,2), pool_type='max', name='pool6')
    
    net = conv_bn_re(data=net,num_filter=250, dropout=0.3,bn=True, relu=True, name_prefix='conv7')
    
    net = conv_bn_re(data=net, num_filter=300, dropout=0.3, relu=True, name_prefix='conv8')
    
    net = conv_bn_re(data=net, num_filter=300, dropout=0.3,bn=True, relu=True, name_prefix='conv9')
    
    net = mx.sym.Pooling(data=net, kernel=(2,2),stride=(2,2), pool_type='max', name='pool9')
    
    net = conv_bn_re(data=net, num_filter=350, dropout=0.4,bn=True, relu=True, name_prefix='conv10')
    
    net = conv_bn_re(data=net, num_filter=350, dropout=0.4, relu=True, name_prefix='conv11')
    
    net = conv_bn_re(data=net, num_filter=400, dropout=0.4, bn=True,relu=True, name_prefix='conv12')
    
    net = mx.sym.Pooling(data=net, kernel=(2,2),stride=(2,2), pool_type='max', name='pool12')
    
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=4096,name='fc1')
    net = mx.sym.Activation(data=net, act_type='relu', name='fc1'+'_relu')
    net = mx.sym.Dropout(data=net,p=0.5)
    
    net = mx.sym.FullyConnected(data=net,num_hidden=900,name='fc2')
    net = mx.sym.Activation(data=net, act_type='relu', name='fc2'+'_relu')
    pred = mx.sym.FullyConnected(data=net, num_hidden=num_class)
    
    pred_ctc = mx.sym.Reshape(data=pred,shape=(-4,seq_len, -1,0))
#    net = mx.sym.Activation(data=net, act_type='relu', name='output'+'_relu')
    label = mx.sym.Variable('label')
    label = mx.sym.Cast(data=label,dtype='int32')
    label = mx.sym.slice(data=label, begin=(0,0),end=(true_batch, num_label))

    loss = mx.sym.contrib.ctc_loss(data=pred_ctc, label=label)
    ctc_loss = mx.sym.MakeLoss(loss)
    
    softmax_class = mx.symbol.SoftmaxActivation(data=pred)
    softmax_loss = mx.sym.MakeLoss(softmax_class)
    softmax_loss = mx.sym.BlockGrad(softmax_loss)
    return mx.sym.Group([softmax_loss, ctc_loss])
    
    
if __name__=='__main__':
    sym = get_symbol(60,40, 40,true_batch=4)
    mx.viz.plot_network(symbol=sym)
#    arg_name = sym.list_arguments()
#    out_name = sym.get_internals().list_outputs()
#    arg_shape, out_shape, _ = sym.get_internals().infer_shape(data=(4,1,32,256))
#    for name, shape in zip(out_name,out_shape):
#        print(name,':',shape)
#    

    
    
    
    
