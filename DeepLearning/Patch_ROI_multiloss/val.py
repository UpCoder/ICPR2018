# -*- coding=utf-8 -*-
from resnet_val import val
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from resnet import inference_small
from image_processing import image_preprocessing
from Net.BaseNet.ResNetMultiPhaseExpand.Config import Config as net_config
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')


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
    filename, expandfilename, label = tf.train.slice_input_producer([filenames, expandfilenames, labels], shuffle=False, num_epochs=1)
    num_process_threads = 1
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

parent_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ROI'
def generate_paths(dir_name, state, target_labels=[0, 1, 2, 3, 4], true_labels=[0, 1, 2, 3, 4]):
    def findSubStr(str, substr, i):
        count = 0
        while i > 0:
            index = str.find(substr)
            if index == -1:
                return -1
            else:
                str = str[index + 1:]
                i -= 1
                count = count + index + 1
        return count - 1
    roi_paths = []
    roi_expand_paths = []
    labels = []
    cur_dir = os.path.join(dir_name, state)
    names = os.listdir(cur_dir)
    for name in names:
        target_label = int(name[-1])
        if target_label not in target_labels:
            # 不是我们训练的目标
            continue
        labels.append(true_labels[target_labels.index(target_label)])
        cur_path = os.path.join(cur_dir, name)
        roi_paths.append(os.path.join(cur_path, 'roi.png'))
        roi_expand_paths.append(os.path.join(cur_path, 'roi_expand.png'))

    return roi_paths, roi_expand_paths, labels

def distorted_inputs(target_labels=[0, 1, 2, 3, 4], true_labels=[0, 1, 2, 3, 4]):
    vals_output = generate_paths(parent_dir, 'val', target_labels, true_labels)
    return distorted_inputs_unit(
               vals_output[0],
               vals_output[1],
               vals_output[2],
               True,
               roi_size=net_config.ROI_SIZE_W,
               roi_expand_size=net_config.EXPAND_SIZE_W)


def main(_):
    images, expand_images, labels = distorted_inputs(
        target_labels=[0, 1, 2, 3], true_labels=[0, 1, 2, 3]
    )
    is_training = tf.placeholder('bool', [], name='is_training')
    logits = inference_small(
        images,
        expand_images,
        phase_names=['NC', 'ART', 'PV'],
        num_classes=4,
        is_training=True,
        )
    roi_outputs = generate_paths(parent_dir, 'val', target_labels=[0, 1, 2, 3, 4], true_labels=[0, 1, 2, 3, 4])
    val(is_training, logits, images, labels, k=1, roi_paths=roi_outputs[0])


if __name__ == '__main__':
    tf.app.run()
