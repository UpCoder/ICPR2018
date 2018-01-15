# -*- coding=utf-8 -*-
import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from config import Config

import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]



activation = tf.nn.relu


def inference(x, expand_x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True):
    c = Config()
    c['bottleneck'] = bottleneck
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    if num_classes != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    return x


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x, x_expand,
                    phase_names,
                    is_training,
                    co_occurrence=False,
                    num_blocks=3, # 6n+2 total weight layers will be used.
                    use_bias=False, # defaults to using batch norm
                    batch_size=None,
                    point_phase=[0, 1, 2],
                    num_classes=10):
    '''
    :param x: Patch的张量 batch*W*H*CHANNEL
    :param phase_names: 对应的三个期项的明知
    :param is_training: 代表是否是训练，主要是batch normalization的时候处理不一样
    :param num_blocks:
    :param use_bias:
    :param batch_size:
    :param point_phase: 指定需要用那几个phase 默认的话是全都使用
    :param num_classes: 最终分类的类别个数
    :return:
    '''
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['num_classes'] = num_classes
    return inference_small_config_pre([x, x_expand], c, phase_names, co_occurrence, batch_size=batch_size, pointed_phase=point_phase)

# ConvNet->reduce_mean->concat->FC
def inference_small_config_lstm(xs_expand, c, phase_names, xs_names=['ROI', 'EXPAND'], batch_size=None, ksize=[3, 3]):
    c['bottleneck'] = False
    c['stride'] = 1
    CONV_OUT = []
    CONV_OUT_index = 0

    NUM_LAYERS = 2
    for xs_index, xs in enumerate(xs_expand):
        with tf.variable_scope(xs_names[xs_index]):
            for index, phase_name in (enumerate(phase_names)):
                c['ksize'] = ksize[xs_index]
                x = xs[:, :, :, index]
                x = tf.expand_dims(
                    x,
                    dim=3
                )
                with tf.variable_scope(phase_name):
                    with tf.variable_scope('scale1'):
                        c['conv_filters_out'] = 16
                        c['block_filters_internal'] = 16
                        c['stack_stride'] = 1
                        x = conv(x, c)
                        x = bn(x, c)
                        x = activation(x)
                        x = stack(x, c)

                    with tf.variable_scope('scale2'):
                        c['block_filters_internal'] = 32
                        c['stack_stride'] = 2
                        x = stack(x, c)

                    with tf.variable_scope('scale3'):
                        c['block_filters_internal'] = 64
                        c['stack_stride'] = 2
                        x = stack(x, c)
                    # post-net
                    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
                    CONV_OUT.append(x)
                    CONV_OUT_index += 1
                    print CONV_OUT
    HIDDEN_SIZE = CONV_OUT[0].get_shape().as_list()[1]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
    if batch_size is None:
        initial_state = cell.zero_state(xs_expand[0].get_shape().as_list()[0], tf.float32)
    else:
        initial_state = cell.zero_state(batch_size, tf.float32)
    state = initial_state
    outputs = []
    with tf.variable_scope('RNN_LSTM'):
        for time_step in range(len(CONV_OUT)):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            print CONV_OUT[time_step]
            print state
            cell_output, state = cell(CONV_OUT[time_step], state)
            outputs.append(cell_output)
    LSTM_OUTPUT = None
    for i in range(len(outputs)):
        if LSTM_OUTPUT is None:
            LSTM_OUTPUT = outputs[i]
        else:
            LSTM_OUTPUT = tf.concat([LSTM_OUTPUT, outputs[i]], axis=1)
    # outputs = outputs[-1]   # 只需要最后的输出即可？
    outputs = LSTM_OUTPUT
    print 'final fc input is ', outputs
    if c['num_classes'] != None:
        print 'before fc layers, the dimension: ', outputs
        with tf.variable_scope('fc'):
            x = fc(outputs, c)
    print 'x is ', x
    return x

def inference_small_config_bilstm(xs_expand, c, phase_names, xs_names=['ROI', 'EXPAND'], batch_size=None, ksize=[3, 3]):
    c['bottleneck'] = False
    c['stride'] = 1
    CONV_OUT = []
    CONV_OUT_index = 0

    NUM_LAYERS = 2
    for xs_index, xs in enumerate(xs_expand):
        with tf.variable_scope(xs_names[xs_index]):
            for index, phase_name in (enumerate(phase_names)):
                c['ksize'] = ksize[xs_index]
                x = xs[:, :, :, index]
                x = tf.expand_dims(
                    x,
                    dim=3
                )
                with tf.variable_scope(phase_name):
                    with tf.variable_scope('scale1'):
                        c['conv_filters_out'] = 16
                        c['block_filters_internal'] = 16
                        c['stack_stride'] = 1
                        x = conv(x, c)
                        x = bn(x, c)
                        x = activation(x)
                        x = stack(x, c)

                    with tf.variable_scope('scale2'):
                        c['block_filters_internal'] = 32
                        c['stack_stride'] = 2
                        x = stack(x, c)

                    with tf.variable_scope('scale3'):
                        c['block_filters_internal'] = 64
                        c['stack_stride'] = 2
                        x = stack(x, c)
                    # post-net
                    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
                    CONV_OUT.append(x)
                    CONV_OUT_index += 1
                    print CONV_OUT
    HIDDEN_SIZE = CONV_OUT[0].get_shape().as_list()[1]
    lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * NUM_LAYERS)
    if batch_size is None:
        initial_state = fw_cell.zero_state(xs_expand[0].get_shape().as_list()[0], tf.float32)
    else:
        initial_state = fw_cell.zero_state(batch_size, tf.float32)
    state = initial_state
    outputs_fw = []
    with tf.variable_scope('RNN_LSTM_FW'):
        for time_step in range(len(CONV_OUT)):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            print CONV_OUT[time_step]
            print state
            cell_output, state = fw_cell(CONV_OUT[time_step], state)
            outputs_fw.append(cell_output)

    lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * NUM_LAYERS)
    if batch_size is None:
        initial_state = fw_cell.zero_state(xs_expand[0].get_shape().as_list()[0], tf.float32)
    else:
        initial_state = fw_cell.zero_state(batch_size, tf.float32)
    state = initial_state
    outputs_bw = []
    with tf.variable_scope('RNN_LSTM_BW'):
        for time_step in range(len(CONV_OUT)):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            print CONV_OUT[time_step]
            print state
            cell_output, state = bw_cell(CONV_OUT[len(CONV_OUT) - time_step - 1], state)
            outputs_bw.append(cell_output)


    LSTM_OUTPUT = None
    for i in range(len(outputs_fw)):
        if LSTM_OUTPUT is None:
            LSTM_OUTPUT = outputs_fw[i]
        else:
            LSTM_OUTPUT = tf.concat([LSTM_OUTPUT, outputs_fw[i]], axis=1)
    print 'after fw fc input is ', LSTM_OUTPUT

    for i in range(len(outputs_bw)):
        if LSTM_OUTPUT is None:
            LSTM_OUTPUT = outputs_bw[i]
        else:
            LSTM_OUTPUT = tf.concat([LSTM_OUTPUT, outputs_bw[i]], axis=1)

    # outputs = outputs[-1]   # 只需要最后的输出即可？
    outputs = LSTM_OUTPUT
    print 'after bw fc input is ', LSTM_OUTPUT
    if c['num_classes'] != None:
        print 'before fc layers, the dimension: ', outputs
        with tf.variable_scope('fc'):
            x = fc(outputs, c)
    print 'x is ', x
    return x
# ConvNet->reduce_mean->concat->FC
def inference_small_config_pre(xs_expand, c, phase_names, co_occurrence=False, xs_names=['Patch', 'ROI'], batch_size=None,ksize=[3, 3], pointed_phase=[0, 1, 2]):
    c['bottleneck'] = False
    c['stride'] = 1
    CONV_OUT = None
    result = []
    for xs_index, xs in enumerate(xs_expand):
        with tf.variable_scope(xs_names[xs_index]):
            for index, phase_name in (enumerate(phase_names)):
                c['ksize'] = ksize[xs_index]
                if not co_occurrence:
                    # parallel的情况，可以任意的选择phase的个数
                    if index not in pointed_phase:
                        continue
                    x = xs[:, :, :, index]
                    x = tf.expand_dims(
                        x,
                        dim=3
                    )
                else:

                    if index != 0:
                        # 共生的情况下，我们只需要计算一次即可
                        continue
                    x = xs
                with tf.variable_scope(phase_name):
                    with tf.variable_scope('scale1'):
                        c['conv_filters_out'] = 16
                        c['block_filters_internal'] = 16
                        c['stack_stride'] = 1
                        x = conv(x, c)
                        x = bn(x, c)
                        x = activation(x)
                        x = stack(x, c)

                    with tf.variable_scope('scale2'):
                        c['block_filters_internal'] = 32
                        c['stack_stride'] = 2
                        x = stack(x, c)

                    with tf.variable_scope('scale3'):
                        c['block_filters_internal'] = 64
                        c['stack_stride'] = 2
                        x = stack(x, c)
                    # post-net
                    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
                    with tf.variable_scope(xs_names[xs_index] + 'fc'):
                        result.append(fc(x, c))
                    if CONV_OUT is None:
                        CONV_OUT = x
                    else:
                        CONV_OUT = tf.concat([CONV_OUT, x], axis=1)
                    print CONV_OUT
    print 'final fc input is ', CONV_OUT
    if c['num_classes'] != None:
        print 'before fc layers, the dimension: ', CONV_OUT
        with tf.variable_scope('fc'):
            x = fc(CONV_OUT, c)
            result.append(x)
    return result


# ConvNet->reduce_mean->FC->concat->FC
def inference_small_config(xs_expand, c, phase_names, xs_names=['ROI', 'EXPAND'], ksize=[3, 3]):
    c['bottleneck'] = False
    c['stride'] = 1
    CONV_OUT = None
    for xs_index, xs in enumerate(xs_expand):
        with tf.variable_scope(xs_names[xs_index]):
            for index, phase_name in (enumerate(phase_names)):
                c['ksize'] = ksize[xs_index]
                x = xs[:, :, :, index]
                x = tf.expand_dims(
                    x,
                    dim=3
                )
                with tf.variable_scope(phase_name):
                    with tf.variable_scope('scale1'):
                        c['conv_filters_out'] = 16
                        c['block_filters_internal'] = 16
                        c['stack_stride'] = 1
                        x = conv(x, c)
                        x = bn(x, c)
                        x = activation(x)
                        x = stack(x, c)

                    with tf.variable_scope('scale2'):
                        c['block_filters_internal'] = 32
                        c['stack_stride'] = 2
                        x = stack(x, c)

                    with tf.variable_scope('scale3'):
                        c['block_filters_internal'] = 64
                        c['stack_stride'] = 2
                        x = stack(x, c)
                    # post-net
                    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
                    if c['num_classes'] != None:
                        print 'before fc layers, the dimension: ', x
                        with tf.variable_scope('fc'):
                            x = fc(x, c)
                    if CONV_OUT is None:
                        CONV_OUT = x
                    else:
                        CONV_OUT = tf.concat([CONV_OUT, x], axis=1)
                    print CONV_OUT
    print 'final fc input is ', CONV_OUT
    if c['num_classes'] != None:
        with tf.variable_scope('fc'):
            x = fc(CONV_OUT, c)
    return x


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(3, 3, rgb * 255.0)
    bgr = tf.concat(3, [blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr

'''
    自定义的loss函数
    尽量避免将0错分为1的这种情况，所以我们当错误的把0分为1的生时候，我们要增加它的权重
    :param logits softmax的输出 one hot 编码
    :param labels 真实值 one hot 编码
'''
def loss_self(logits, labels):
    logits = tf.nn.softmax(logits)
    # 当分类正确的时候，应该是接近0的，log(1)=0
    # 当分类错误的时候，应该是偏大的， log(0)=正无穷
    loss_zero = tf.log(tf.clip_by_value(logits[:, 0], 1e-10, 1)) * labels[:, 0] * 2.0
    loss_one = tf.log(tf.clip_by_value(logits[:, 1], 1e-10, 1)) * labels[:, 1]
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_mean = -tf.reduce_mean(loss_zero + loss_one)
    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    return loss_


def loss(logits, labels, name='loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar(name, loss_)

    return loss_


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x

def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
if __name__ == '__main__':
    roi_tensor = tf.placeholder(tf.float32,[30, 64, 64, 3])
    expand_tensor = tf.placeholder(tf.float32, [30, 64, 64, 3])
    c = Config()
    c['is_training'] = tf.convert_to_tensor(False,
                                            dtype='bool',
                                            name='is_training')
    num_blocks = 3  # 6n+2 total weight layers will be used.
    use_bias = False  # defaults to using batch norm
    c['use_bias'] = use_bias
    c['fc_units_out'] = 4
    c['num_blocks'] = num_blocks
    c['num_classes'] = 4
    # inference_small_config_bilstm([roi_tensor, expand_tensor], c, ['NC', 'ART', 'PV'])
    logits = inference_small(
        roi_tensor,
        expand_tensor,
        co_occurrence=True,
        phase_names=['NC', 'ART', 'PV'],
        num_classes=4,
        is_training=True,
        point_phase=[2]
    )
    print logits