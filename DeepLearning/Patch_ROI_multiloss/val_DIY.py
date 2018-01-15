# -*- coding=utf-8 -*-
import tensorflow as tf
import os
import sys
from load_liver_density import load_raw_liver_density
import numpy as np
from resnet import inference_small
from Config import Config as net_config
from utils.Tools import shuffle_image_label, read_mhd_image, get_boundingbox, convert2depthlaster, calculate_acc_error
from PIL import Image
from glob import glob
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")

def load_patch(patch_path, return_roi=False, parent_dir=None):
    if not return_roi:
        if patch_path.endswith('.jpg'):
            return Image.open(patch_path)
        if patch_path.endswith('.npy'):
            return np.load(patch_path)
    else:
        phasenames = ['NC', 'ART', 'PV']
        if patch_path.endswith('.jpg'):
            basename = os.path.basename(patch_path)
            basename = basename[: basename.rfind('_')]
            mask_images = []
            mhd_images = []
            for phasename in phasenames:
                image_path = glob(os.path.join(parent_dir, basename, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(parent_dir, basename, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path, rejust=True)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                # xmin -= 15
                # xmax += 15
                # ymin -= 15
                # ymax += 15
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
            mhd_images = convert2depthlaster(mhd_images)
            return mhd_images
        if patch_path.endswith('.npy'):
            basename = os.path.basename(patch_path)
            basename = basename[: basename.rfind('_')]
            mask_images = []
            mhd_images = []
            for phasename in phasenames:
                image_path = glob(os.path.join(parent_dir, basename, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(parent_dir, basename, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path, rejust=False)    # 因为存储的是ｎｐｙ格式，所以不进行窗宽窗位的调整
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                # xmin -= 15
                # xmax += 15
                # ymin -= 15
                # ymax += 15
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
            mhd_images = convert2depthlaster(mhd_images)
            return mhd_images
def resize_images(images, size, rescale=True):
    res = np.zeros(
        [
            len(images),
            size,
            size,
            3
        ],
        np.float32
    )
    for i in range(len(images)):
        img = Image.fromarray(np.asarray(images[i], np.uint8))
        # data augment
        random_int = np.random.randint(0, 4)
        img = img.rotate(random_int * 90)
        random_int = np.random.randint(0, 2)
        if random_int == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        random_int = np.random.randint(0, 2)
        if random_int == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        img = img.resize([size, size])
        if rescale:
            res[i, :, :, :] = np.asarray(img, np.float32) / 255.0
            res[i, :, :, :] = res[i, :, :, :] - 0.5
            res[i, :, :, :] = res[i, :, :, :] * 2.0
        else:
            res[i, :, :, :] = np.asarray(img, np.float32)
    return res
def main(_):
    roi_images = tf.placeholder(
        shape=[
            None,
            net_config.ROI_SIZE_W,
            net_config.ROI_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='roi_input'
    )
    expand_roi_images = tf.placeholder(
        shape=[
            None,
            net_config.EXPAND_SIZE_W,
            net_config.EXPAND_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='expand_roi_input'
    )
    batch_size_tensor = tf.placeholder(dtype=tf.int32, shape=[])
    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[])
    logits = inference_small(
        roi_images,
        expand_roi_images,
        phase_names=['NC', 'ART', 'PV'],
        num_classes=4,
        point_phase=[2],
        is_training=is_training_tensor,
        batch_size=batch_size_tensor
        )
    # model_path = '/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patch_ROI/models/300.0/'
    # model_path = '/home/give/PycharmProjects/MedicalImage/Net/forpatch/cross_validation/model/multiscale/parallel/0/2200.0'
    model_path = '/home/give/PycharmProjects/ICPR2018/DeepLearning/Patch_ROI_multiloss/models'
    predictions = tf.nn.softmax(logits[2])
    saver = tf.train.Saver(tf.all_variables())
    print predictions

    predicted_label_tensor = tf.argmax(predictions, axis=1)
    print predicted_label_tensor
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    latest = tf.train.latest_checkpoint(model_path)
    if not latest:
        print "No checkpoint to continue from in", model_path
        sys.exit(1)
    print "resume", latest
    saver.restore(sess, latest)

    data_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/only-patch-7/test'
    labels = []
    paths = []
    mapping_label = {0: 0, 1: 1, 2: 2, 3:3}
    for typeid in [0, 1, 2, 3]:
        cur_path = os.path.join(data_dir, str(typeid))
        names = os.listdir(cur_path)
        labels.extend([mapping_label[typeid]] * len(names))
        paths.extend([os.path.join(cur_path, name) for name in names])
    paths, labels = shuffle_image_label(paths, labels)
    start_index = 0
    predicted_labels = []
    liver_density = load_raw_liver_density()
    while True:
        if start_index >= len(paths):
            break
        print start_index, len(paths)
        end_index = start_index + net_config.BATCH_SIZE
        cur_paths = paths[start_index: end_index]
        cur_roi_images = [np.asarray(load_patch(path)) for path in cur_paths]
        cur_expand_roi_images = [
            np.asarray(load_patch(path, return_roi=True, parent_dir='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/val')) for path in
            cur_paths]
        cur_roi_images = resize_images(cur_roi_images, net_config.ROI_SIZE_W, True)
        cur_expand_roi_images = resize_images(cur_expand_roi_images, net_config.EXPAND_SIZE_W, True)
        # cur_liver_densitys = [liver_density[os.path.basename(path)[:os.path.basename(path).rfind('_')]] for
        #                       path in cur_paths]
        # for i in range(len(cur_roi_images)):
        #     for j in range(3):
        #         cur_roi_images[i, :, :, j] = (1.0 * cur_roi_images[i, :, :, j]) / (1.0 * cur_liver_densitys[i][j])
        #         cur_expand_roi_images[i, :, :, j] = (1.0 * cur_expand_roi_images[i, :, :, j]) / (
        #         1.0 * cur_liver_densitys[i][j])
        predicted_batch_labels = sess.run(predicted_label_tensor, feed_dict={
            roi_images: cur_roi_images,
            expand_roi_images: cur_expand_roi_images,
            is_training_tensor: False,
            batch_size_tensor: len(cur_roi_images)
        })
        batch_labels = labels[start_index: end_index]
        predicted_labels.extend(predicted_batch_labels)
        start_index = end_index
        calculate_acc_error(predicted_batch_labels, batch_labels)
    calculate_acc_error(predicted_labels, labels)
if __name__ == '__main__':
    tf.app.run()
