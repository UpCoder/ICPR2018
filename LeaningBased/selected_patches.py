# -*- coding=utf-8 -*-
import os
from utils.Tools import shuffle_array
import shutil
import numpy as np
from multiprocessing import Pool


def split_array(arr, num):
    '''
    将一个数组拆分成多个数组
    :param arr: 待拆分的数组
    :param num: 需要拆分成多少个
    :return:
    '''
    result = []
    length = len(arr)
    pre_num = length / num
    for i in range(num):
        if i != (num-1):
            cur_group = arr[i*pre_num: (i+1)*pre_num]
        else:
            cur_group = arr[i*pre_num: length]
        result.append(cur_group)
    return result


def flatten_arr(arr):
    '''
    将一个数组展评
    :param arr:[7,7,3]的数组
    :return: 第一个ｃｈａｎｎｅｌ的７，７然后是第二个ｃｈａｎｎｅｌ的７，７　以此类推，直到最后一个ｃｈａｎｎｅｌ的７　７
    '''
    res = []
    shape = list(np.shape(arr))
    for i in range(shape[2]):
        res.extend(np.array(arr[:, :, i]).flatten())
    return res


def read_single_process(cur_group):
    '''
    读取一个数组里面的所有文件，使用一个进程
    :param cur_group:　文件路径的数组
    :return:
    '''
    data_group = []
    for path in cur_group:
        patch = np.load(path)
        patch = flatten_arr(patch)
        data_group.append(patch)
    return data_group


def return_patches(source_dir, pre_class_num):
    '''
    读取该目录下面的所有文件, 使用多个进程
    :param source_dir:
    :param pre_class_num:
    :return:
    '''
    names = os.listdir(source_dir)
    names = shuffle_array(names)
    names = names[:pre_class_num]
    paths = [os.path.join(source_dir, name) for name in names]
    process_num = 5     # the number of process
    groups = split_array(paths, process_num)
    for i in range(process_num):
        print len(groups[i])
    pool = Pool()
    results = []
    for i in range(process_num):
        result = pool.apply_async(read_single_process, args=(groups[i],))
        results.append(result)
    patches = []
    for i in range(process_num):
        patches.extend(results[i].get())
    pool.close()
    return patches

def return_patches_multidir(data_dir, subclass_names=['train', 'test'], target_label=[0, 1, 2, 3], pre_class_num=30000):
    '''
    读取多个目录下面的文件
    :param data_dir:
    :param subclass_names: 子目录名称
    :param target_label: 子子目录名称
    :return:
    '''
    patches = []
    for subclass in subclass_names:
        for typeid in target_label:
            patches_oneclass = return_patches(
                os.path.join(data_dir, subclass, str(typeid)),
                pre_class_num
            )
            patches.extend(patches_oneclass)
    return patches

def selected_patches(source_dir, target_dir, patches_num = 30000):
    names = os.listdir(source_dir)
    names = shuffle_array(names)
    names = names[:patches_num]
    paths = [os.path.join(source_dir, name) for name in names]
    for path in paths:
        shutil.copy(path, os.path.join(target_dir, os.path.basename(path)))


def selected_patches_multidir():
    data_dir = '/home/give/Documents/dataset/ICPR2018/dual-dict/boundary'
    target_dir = '/home/give/Documents/dataset/ICPR2018/dual-dict/boundary_balance'
    for subclass in ['train', 'val']:
        for typeid in [0, 1, 2, 3]:
            selected_patches(
                os.path.join(data_dir, subclass, str(typeid)),
                os.path.join(target_dir, subclass, str(typeid))
            )

if __name__ == '__main__':
    patches = return_patches_multidir('/home/give/Documents/dataset/ICPR2018/BoVW-MI/patches')
    print np.shape(patches)