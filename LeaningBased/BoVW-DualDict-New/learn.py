import numpy.random as rn
from numpy import array, zeros, dot
import numpy as np
import scipy.io as scio
import os
from sklearn.cluster import KMeans
from sklearn.externals import joblib


def selected_patches(data_path, limited_num=-1):
    patches_dict = {}
    data = scio.loadmat(data_path)
    for key in data.keys():
        if key.startswith('__'):
            continue
        category_data = data[key]
        if limited_num == -1:
            limited_num = len(category_data)
        else:
            limited_num = min(limited_num, len(category_data))
        shuffed_index = range(len(category_data))
        np.random.shuffle(shuffed_index)
        patches = category_data[shuffed_index]
        patches = patches[:limited_num]
        patches_dict[key] = patches
    return patches_dict


def generate_dictionary(data_dir='/home/give/Documents/dataset/ICPR2018/BoVW-DualDict/data.mat', vocabulary_size=256, limited_num=30000, save_path='./vocabulary.model'):
    patches_dict = selected_patches(data_dir, limited_num=limited_num)
    patches = []
    labeles = []
    for key in patches_dict.keys():
        patches.extend(patches_dict[key])
        labeles.extend([int(key)] * len(patches_dict[key]))
    print 'the patches shape is ', np.shape(patches), ' in ', data_dir
    kmeans_obj = KMeans(n_clusters=vocabulary_size, max_iter=500, n_jobs=8).fit(patches)

    cluster_centroid_objs = kmeans_obj.cluster_centers_
    print 'vocabulary shape is ', np.shape(cluster_centroid_objs), ' in ', data_dir
    joblib.dump(kmeans_obj, save_path)
    np.save(os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0] + '.npy'), kmeans_obj.cluster_centers_)
    return cluster_centroid_objs


if __name__ == "__main__":
    generate_dictionary(data_dir='/home/give/Documents/dataset/ICPR2018/BoVW-DualDict/boundary_data.mat',
                        save_path='./vocabulary_boundary.model', vocabulary_size=128)
    generate_dictionary(data_dir='/home/give/Documents/dataset/ICPR2018/BoVW-DualDict/interior_data.mat',
                        save_path='./vocabulary_interior.model', vocabulary_size=32)
