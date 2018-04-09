# -*- coding=utf-8 -*-
import os
import numpy as np
from sklearn.cluster import KMeans
import scipy.io as scio


def learn_dict(patch_path, limited, dict_size, dict_save_path):
    '''
    学习字典
    :param patch_path:得到patch的path
    :param limited: 每个类别patch的个数
    :param dict_size: 字典的大小
    :param dict_save_path: 字典的存储路径，如果为none，则返回字典
    :return:
    '''
    data = scio.loadmat(patch_path)
    allpatches = []
    for i in data.keys():
        if i.startswith('__'):
            continue
        patches = data[i]
        print np.shape(patches)
        indexs = range(len(patches))
        np.random.shuffle(indexs)
        patches = patches[indexs[:limited]]
        allpatches.extend(patches)
    print np.shape(allpatches)
    kmeans_obj = KMeans(n_clusters=dict_size, n_jobs=8, max_iter=500).fit(allpatches)
    dictionary = kmeans_obj.cluster_centers_
    dictionary = np.array(dictionary)
    if dict_save_path is not None:
        np.save(dict_save_path, dictionary)
    else:
        return dictionary


if __name__ == '__main__':
    learn_dict('/home/give/Documents/dataset/ICPR2018/BoVW/data.mat', 30000, 128, './dictionary-128-30000.npy')