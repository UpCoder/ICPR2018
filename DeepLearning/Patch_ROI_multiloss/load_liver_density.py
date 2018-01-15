# -*- coding=utf-8 -*-
import os
import scipy.io as scio
from glob import glob
import numpy as np


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