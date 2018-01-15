# -*- coding=utf-8 -*-
from resnet_train import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from resnet import inference_small
from image_processing import image_preprocessing
from Net.forpatch.ResNetMultiPhaseExpand.Config import Config as net_config
from Tools import shuffle_image_label

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")

def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames


def distorted_inputs_unit(
        roi_paths,
        roi_expand_paths,
        labels,
        trainable,
        roi_size=128,
        roi_expand_size=256):
    filenames = roi_paths
    expandfilenames = roi_expand_paths
    labels = labels
    filename, expandfilename, label = tf.train.slice_input_producer([filenames, expandfilenames, labels], shuffle=False)
    num_process_threads = 4
    images_and_labels = []
    for thread_id in range(num_process_threads):
        image_buffer = tf.read_file(filename)

        bbox = []
        image = image_preprocessing(
            image_buffer,
            bbox=bbox,
            train=trainable,
            thread_id=thread_id,
            image_size=roi_size
        )
        # image = tf.image.rgb_to_hsv(image)

        expand_image_buffer = tf.read_file(expandfilename)
        bbox = []
        expandimage = image_preprocessing(
            expand_image_buffer,
            bbox=bbox,
            train=trainable,
            thread_id=thread_id,
            image_size=roi_expand_size
        )

        images_and_labels.append([image, expandimage, label])
    batch_image, batch_expand_image, batch_label = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2*num_process_threads*FLAGS.batch_size
    )

    images = tf.cast(batch_image, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, roi_size, roi_size, 3])

    expand_images = tf.cast(batch_expand_image, tf.float32)
    # print expand_images
    expand_images = tf.reshape(expand_images, shape=[FLAGS.batch_size, roi_expand_size, roi_expand_size, 3])
    return images, expand_images, tf.reshape(batch_label, [FLAGS.batch_size])


parent_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases'
def generate_paths(dir_name, state, target_labels=[0, 1, 2, 3]):
    '''
    返回dirname中的所有病灶图像的路径
    :param dir_name:  父文件夹的路径
    :param state: 状态，一般来说父文件夹有两个状态 train 和val
    :param target_labels: 需要文件标注的label
    :return:
    '''
    roi_paths = []
    roi_expand_paths = []
    labels = []

    cur_dir = os.path.join(dir_name, state)
    # names = os.listdir(cur_dir)
    for target_label in target_labels:
        type_dir = os.path.join(cur_dir, str(target_label))
        type_names = os.listdir(type_dir)
        roi_paths.extend([os.path.join(type_dir, name) for name in type_names])
        labels.extend([target_label] * len(type_names))
    roi_paths, labels = shuffle_image_label(roi_paths, labels)
    return roi_paths, roi_paths, labels


def distorted_inputs(target_labels=[0, 1, 2, 3]):
    trains_output = generate_paths(parent_dir, 'train', target_labels)
    vals_output = generate_paths(parent_dir, 'val', target_labels)
    return distorted_inputs_unit(
        trains_output[0],
        trains_output[1],
        trains_output[2],
        True,
        roi_size=net_config.ROI_SIZE_W,
        roi_expand_size=net_config.EXPAND_SIZE_W), \
           distorted_inputs_unit(
               vals_output[0],
               vals_output[1],
               vals_output[2],
               False,
               roi_size=net_config.ROI_SIZE_W,
               roi_expand_size=net_config.EXPAND_SIZE_W)


def main(_):
    [train_images, train_expand_images, train_labels], [val_images, val_expand_images, val_labels] = distorted_inputs(
        target_labels=[0, 1, 2, 3]
    )
    # print train_images
    is_training = tf.placeholder('bool', [], name='is_training')
    images, expand_images, labels = tf.cond(is_training,
                             lambda: (train_images, train_expand_images, train_labels),
                             lambda: (val_images, val_expand_images, val_labels))
    # with tf.Session() as sess:
    #     tf.train.start_queue_runners(sess=sess)
    #     one_hot_label = tf.one_hot(tf.cast(labels, tf.uint8), depth=5)
    #
    #     print sess.run(labels, feed_dict={is_training: True})
    #     print np.shape(sess.run(one_hot_label, feed_dict={is_training: True}))
    print labels
    logits = inference_small(
        images,
        expand_images,
        phase_names=['NC', 'ART', 'PV'],
        num_classes=4,
        is_training=True,
        )
    print labels
    save_model_path = '/home/give/PycharmProjects/MedicalImage/Net/forpatch/ResNetMultiPhaseExpand/models'
    train(is_training, logits, images, expand_images, labels, save_model_path=save_model_path, step_width=100)


if __name__ == '__main__':
    tf.app.run()
