# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from config import Config

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op


activation = tf.nn.relu


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x, x_expand,
                    phase_names,
                    is_training,
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
    return inference_small_config_pre([x, x_expand], c, phase_names, batch_size=batch_size, pointed_phase=point_phase)


# ConvNet->reduce_mean->concat->FC
def inference_small_config_pre(xs_expand, c, phase_names, xs_names=['Patch', 'ROI'], batch_size=None,ksize=[3, 3], pointed_phase=[0, 1, 2]):
    c['bottleneck'] = False
    c['stride'] = 1
    CONV_OUT = None
    for xs_index, xs in enumerate(xs_expand):
        with tf.variable_scope(xs_names[xs_index]):
            for index, phase_name in (enumerate(phase_names)):
                if index not in pointed_phase:
                    continue
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
    return x


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
    roi_tensor = tf.placeholder(tf.float32, [30, 64, 64, 3])
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
    inference_small_config_pre([roi_tensor, expand_tensor], c, ['NC', 'ART', 'PV'])