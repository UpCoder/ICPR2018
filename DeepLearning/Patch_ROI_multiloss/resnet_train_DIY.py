# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
from utils.Tools import changed_shape, calculate_acc_error, acc_binary_acc, shuffle_image_label, read_mhd_image, get_boundingbox, convert2depthlaster
from glob import glob
import shutil
import scipy.io as scio
from Config import Config as net_config
from PIL import Image

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('load_model_path',
                           '/home/give/PycharmProjects/MedicalImage/Net/ICIP/Patched/models/1200.0',
                           '''the model reload path''')
tf.app.flags.DEFINE_string('save_model_path', './models', 'the saving path of the model')
tf.app.flags.DEFINE_string('log_dir', './log/train',
                           """The Summury output directory""")
tf.app.flags.DEFINE_string('log_val_dir', './log/val',
                           """The Summury output directory""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
tf.app.flags.DEFINE_integer('max_steps', 10000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')

loss_local_coefficient = 0.25
loss_global_coefficient = 0.25
loss_all_coefficient = 0.5

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
                # print os.path.join(parent_dir, basename, phasename + '_Image*.mhd')
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
def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def calculate_accuracy(logits, labels, arg_index=1):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, arg_index), tf.arg_max(labels, arg_index))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy

class DataSet:
    @staticmethod
    def load_liver_density(data_dir='/home/give/PycharmProjects/MedicalImage/Net/forpatch/ResNetMultiPhaseExpand'):
        '''
        加载调整过窗宽窗位的肝脏平均密度
        :param data_dir: mat文件的路径
        :return:dict类型的对象，key是我们的文件名，value是长度为３的数组代表的是三个Ｐｈａｓｅ的平均密度
        '''
        mat_paths = glob(os.path.join(data_dir, 'liver_density*.mat'))
        total_liver_density = {}
        for mat_path in mat_paths:
            liver_density = scio.loadmat(mat_path)
            for (key, value) in liver_density.items():
                if key.startswith('__'):
                    continue
                if key in total_liver_density.keys():
                    print 'Error', key
                total_liver_density[key] = np.array(value).squeeze()
        return total_liver_density
    @staticmethod
    def load_raw_liver_density(data_dir='/home/give/PycharmProjects/MedicalImage/Net/ICIP/Patched'):
        '''
        加载原生的肝脏平均密度
        :param data_dir: mat文件的路径
        :return:dict类型的对象，key是我们的文件名，value是长度为３的数组代表的是三个Ｐｈａｓｅ的平均密度
        '''
        mat_paths = glob(os.path.join(data_dir, 'raw_liver_density*.mat'))
        total_liver_density = {}
        for mat_path in mat_paths:
            liver_density = scio.loadmat(mat_path)
            for (key, value) in liver_density.items():
                if key.startswith('__'):
                    continue
                if key in total_liver_density.keys():
                    print 'Error', key
                total_liver_density[key] = np.array(value).squeeze()
        return total_liver_density
    @staticmethod
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
    @staticmethod
    def generate_paths(dir_name, target_labels=[0, 1, 2, 3], mapping_label = {0:0, 1:1, 2:2, 3:3}, shuffle=True):
        '''
        返回dirname中的所有病灶图像的路径
        :param dir_name:  父文件夹的路径
        :param cross_ids: 包含的交叉的折，一般来说我们做三折交叉验证,cross_ids就是[0, 1] 或者是[2]
        :param target_labels: 需要文件标注的label
        :return:
        '''
        roi_paths = []
        roi_expand_paths = []
        labels = []
        cur_dir = dir_name
        print cur_dir
        # names = os.listdir(cur_dir)
        for target_label in target_labels:
            type_dir = os.path.join(cur_dir, str(target_label))
            type_names = os.listdir(type_dir)
            roi_paths.extend([os.path.join(type_dir, name) for name in type_names])
            labels.extend([mapping_label[target_label]] * len(type_names))
        if shuffle:
            roi_paths, labels = shuffle_image_label(roi_paths, labels)
        return roi_paths, roi_paths, labels

    def __init__(self, data_dir, state, pre_load=False, divied_liver=False, rescale=True, expand_is_roi=False, full_roi_path=None):
        '''
        DataSet的初始化函数
        :param data_dir: 数据的文件夹，存储ｐａｔｃｈ的路径　一般结构是data_dir/train/0/*
        :param state:
        :param pre_load: 是否将数据全部提前加载进来, 默认是Ｆａｌｓｅ
        :param divied_liver: 是否除以肝脏的平均密度
        :param rescale: 是否将像素值进行放缩，放缩到[-1, 1]之间
        :param full_roi_path: 完整的存储ｒｏｉ的路径，不是存储ｐａｔｃｈ的路径
        :param expand_is_roi: 决定ｅｘｐａｎｄ是否是ＲＯＩ　如果是则ｅｘｐａｎｄ代表的是完整的ＲＯＩ，否则ｅｘｐａｎｄ就是同一个ｐａｔｃｈ放缩的不同的ｓｃａｌｅ
        '''
        self.roi_paths, self.expand_roi_path, self.labels = DataSet.generate_paths(
            data_dir,
        )
        if self.roi_paths[0].endswith('.npy'):
            self.using_raw = True
        else:
            self.using_raw = False
        self.state = state
        self.epoch_num = 0
        self.start_index = 0
        self.liver_density = DataSet.load_liver_density()   # 调整过窗宽窗位的肝脏平均密度
        self.raw_liver_density = DataSet.load_raw_liver_density()   # 原始的像素值的平均密度
        print self.raw_liver_density
        self.divied_liver = divied_liver
        self.rescale = rescale
        self.full_roi_path=full_roi_path
        self.expand_is_roi = expand_is_roi
    def get_next_batch(self, batch_size):
        while True:
            cur_roi_paths = []
            cur_expand_roi_paths = []
            cur_labels = []
            end_index = self.start_index + batch_size
            if end_index > len(self.roi_paths):
                self.epoch_num += 1
                cur_roi_paths.extend(self.roi_paths[self.start_index: len(self.roi_paths)])
                cur_roi_paths.extend(self.roi_paths[:end_index - len(self.roi_paths)])

                cur_expand_roi_paths.extend(self.expand_roi_path[self.start_index: len(self.roi_paths)])
                cur_expand_roi_paths.extend(self.expand_roi_path[:end_index - len(self.roi_paths)])

                cur_labels.extend(self.labels[self.start_index: len(self.roi_paths)])
                cur_labels.extend(self.labels[:end_index - len(self.roi_paths)])
                self.start_index = end_index - len(self.roi_paths)
                print 'state: ', self.state, ' epoch: ', self.epoch_num

            else:
                cur_roi_paths.extend(self.roi_paths[self.start_index: end_index])
                cur_expand_roi_paths.extend(self.expand_roi_path[self.start_index: end_index])
                cur_labels.extend(self.labels[self.start_index: end_index])
                self.start_index = end_index

            cur_roi_images = [np.asarray(load_patch(path)) for path in cur_roi_paths]
            cur_expand_roi_images = [np.asarray(load_patch(path, return_roi=self.expand_is_roi, parent_dir=self.full_roi_path)) for path in cur_expand_roi_paths]
            cur_roi_images = DataSet.resize_images(cur_roi_images, net_config.ROI_SIZE_W, self.rescale)
            cur_expand_roi_images = DataSet.resize_images(cur_expand_roi_images, net_config.EXPAND_SIZE_W, self.rescale)
            # print np.shape(cur_roi_images)
            if self.divied_liver:
                if self.using_raw:
                    cur_liver_densitys = [
                        self.raw_liver_density[os.path.basename(path)[:os.path.basename(path).rfind('_')]] for path in
                        cur_roi_paths]
                else:
                    cur_liver_densitys = [self.liver_density[os.path.basename(path)[:os.path.basename(path).rfind('_')]]
                                          for path in cur_roi_paths]
                for i in range(len(cur_roi_images)):
                    for j in range(3):
                        cur_roi_images[i, :, :, j] = cur_roi_images[i, :, :, j] / cur_liver_densitys[i][j]
                        cur_expand_roi_images[i, :, :, j] = cur_expand_roi_images[i, :, :, j] / cur_liver_densitys[i][j]
            yield cur_roi_images, cur_expand_roi_images, cur_labels


def train(logits, images_tensor, expand_images_tensor, labels_tensor, is_training_tensor, save_model_path=None, step_width=100, record_loss=False):
    train_dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/140_only_patch/train', 'train',
                            rescale=True, divied_liver=False, expand_is_roi=True,
                            full_roi_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train')
    val_dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/140_only_patch/val', 'val',
                          rescale=True, divied_liver=False, expand_is_roi=True,
                          full_roi_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train')
    loss_value_record_file_path = '/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patch_ROI/models_draw_line/loss_value'
    if record_loss:
        writed_filed = open(loss_value_record_file_path, 'w')
        writed_filed.write('step training loss, val loss, training acc, val acc\n')
        writed_filed.close()
    train_batchdata = train_dataset.get_next_batch(net_config.BATCH_SIZE)
    val_batchdata = val_dataset.get_next_batch(net_config.BATCH_SIZE)

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    loss_local = loss(logits[0], labels_tensor)
    loss_global = loss(logits[1], labels_tensor)
    loss_all = loss(logits[2], labels_tensor)
    loss_ = loss_local_coefficient * loss_local + loss_global_coefficient * loss_global * loss_all_coefficient * loss_all
    predictions = tf.nn.softmax(logits[2])
    print 'predictions shape is ', predictions
    print 'label is ', labels_tensor
    top1_error = top_k_error(predictions, labels_tensor, 1)
    labels_onehot = tf.one_hot(labels_tensor, logits[2].get_shape().as_list()[-1])
    print 'output node is ', logits[2].get_shape().as_list()[-1]
    accuracy_tensor = calculate_accuracy(predictions, labels_onehot)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.summary.image('images', images_tensor)

        for var in tf.trainable_variables():
            tf.summary.image(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.log_val_dir, sess.graph)
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.load_model_path)
        if not latest:
            import sys
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_, accuracy_tensor]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)
        train_roi_batch_images, train_expand_roi_batch_images, train_labels = train_batchdata.next()
        o = sess.run(i, feed_dict={
            images_tensor: train_roi_batch_images,
            expand_images_tensor: train_expand_roi_batch_images,
            labels_tensor: train_labels,
            is_training_tensor: True
        })

        loss_value = o[1]

        # calculate the value of loss of validation
        if record_loss:
            val_roi_batch_images, val_expand_roi_batch_images, val_labels = val_batchdata.next()
            val_loss_value, val_accuracy_value = sess.run([loss_, accuracy_tensor], feed_dict={
                images_tensor: train_roi_batch_images,
                expand_images_tensor: train_expand_roi_batch_images,
                labels_tensor: train_labels,
                is_training_tensor: False
            })
            printed_str = 'step %d, training loss_value %f accuracy %f, validation loss_value %f accuracy %f'\
                          % (step, loss_value, o[2], val_loss_value, val_accuracy_value)
            print printed_str
            writed_str = '%d %f %f %f %f\n' % (step, loss_value, val_loss_value, o[2], val_accuracy_value)
            writed_filed = open(loss_value_record_file_path, 'a+')
            writed_filed.write(writed_str)
            writed_filed.close()

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if (step - 1) % step_width == 0:
            top1_error_value, accuracy_value, labels_values, predictions_values = sess.run([top1_error, accuracy_tensor, labels_tensor, predictions], feed_dict={
                images_tensor: train_roi_batch_images,
                expand_images_tensor: train_expand_roi_batch_images,
                labels_tensor: train_labels,
                is_training_tensor: True
            })
            predictions_values = np.argmax(predictions_values, axis=1)
            examples_per_sec = FLAGS.batch_size / float(duration)
            # accuracy = eval_accuracy(predictions_values, labels_values)
            format_str = ('step %d, loss = %.2f, top1 error = %g, accuracy value = %g  (%.1f examples/sec; %.3f '
                          'sec/batch)')

            print(format_str % (step, loss_value, top1_error_value, accuracy_value, examples_per_sec, duration))
        if write_summary:
            summary_str = o[3]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % step_width == 0:

            checkpoint_path = os.path.join(save_model_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            save_dir = os.path.join(save_model_path, str(step))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            filenames = glob(os.path.join(save_model_path, '*-'+str(int(step + 1))+'.*'))
            for filename in filenames:
                shutil.copy(
                    filename,
                    os.path.join(save_dir, os.path.basename(filename))
                )
        # Run validation periodically
        if step > 1 and step % step_width == 0:
            if not record_loss:
                val_roi_batch_images, val_expand_roi_batch_images, val_labels = val_batchdata.next()
            _, top1_error_value, summary_value, accuracy_value, labels_values, predictions_values = sess.run(
                [val_op, top1_error, summary_op, accuracy_tensor, labels_tensor, predictions],
                {
                    images_tensor: val_roi_batch_images,
                    expand_images_tensor: val_expand_roi_batch_images,
                    labels_tensor: val_labels,
                    is_training_tensor: False
                })
            predictions_values = np.argmax(predictions_values, axis=1)
            # accuracy = eval_accuracy(predictions_values, labels_values)
            calculate_acc_error(
                logits=predictions_values,
                label=labels_values,
                show=True
            )
            print('Validation top1 error %.2f, accuracy value %f'
                  % (top1_error_value, accuracy_value))
            val_summary_writer.add_summary(summary_value, step)