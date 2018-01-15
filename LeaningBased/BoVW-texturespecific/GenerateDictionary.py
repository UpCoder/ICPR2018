import os
import numpy as np
from sklearn.cluster import KMeans


def selected_patches(data_dir, set_limited_num=None):
    patches_dict = {}
    limited_num = -1
    for i in range(4):
        data_path = os.path.join(data_dir, str(i), 'data.npy')
        patches = np.load(data_path)
        patches_dict[i] = patches
        if limited_num == -1:
            limited_num = len(patches)
        else:
            limited_num = min(limited_num, len(patches))
    if set_limited_num is not None:
        limited_num = set_limited_num
    for key in patches_dict.keys():
        patches = patches_dict[key]
        shuffed_index = range(len(patches))
        np.random.shuffle(shuffed_index)
        patches = patches[shuffed_index]
        patches = patches[:limited_num]
        patches_dict[key] = patches
    return patches_dict


def generate_dictionary(data_dir, vocabulary_size=128):
    patches_dict = selected_patches(data_dir, set_limited_num=20000)
    patches = []
    labeles = []
    for key in patches_dict.keys():
        patches.extend(patches_dict[key])
        labeles.extend([int(key)] * len(patches_dict[key]))
    print 'the patches shape is ', np.shape(patches), ' in ', data_dir
    kmeans_obj = KMeans(n_clusters=vocabulary_size, n_jobs=8).fit(patches)
    cluster_centroid_objs = kmeans_obj.cluster_centers_
    print 'vocabulary shape is ', np.shape(cluster_centroid_objs), ' in ', data_dir
    np.save(os.path.join(data_dir, 'vocabulary.npy'), cluster_centroid_objs)


def generate_dictionary_multidir(data_dir, vocabulary_size=128):
    names = os.listdir(data_dir)
    for name in names:
        generate_dictionary(os.path.join(data_dir, name), vocabulary_size)

if __name__ == '__main__':
    generate_dictionary_multidir('/home/give/Documents/dataset/ICPR2018/BoVW-TextSpecific')
