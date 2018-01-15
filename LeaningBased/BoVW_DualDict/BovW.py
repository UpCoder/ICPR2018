# -*- coding=utf-8 -*-
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from glob import glob
from utils.Tools import read_mhd_image, get_boundingbox, convert2depthlaster, image_erode, image_expand, calculate_acc_error
import scipy.io as scio
from multiprocessing import Pool
from shutil import copy
phasenames=['NC', 'ART', 'PV']
patch_size = 8
divided_liver = True
category_number = 4


def load_raw_liver_density(data_dir='/home/give/PycharmProjects/MedicalImage/liver-density'):
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
liver_density_dict = load_raw_liver_density()
def load_patch(patch_path):
    if patch_path.endswith('.jpg'):
        return Image.open(patch_path)
    if patch_path.endswith('.npy'):
        return np.load(patch_path)

def generate_density_feature(data_dir):
    print 'extracting features start from ', data_dir
    names = os.listdir(data_dir)
    features = []
    target_label = int(data_dir[-1])
    np.random.shuffle(names)
    # if target_label != 0:
    #     if len(names) > 5000:
    #         names = names[:5000]
    if target_label == 0:
        names.extend(names)
    for name in names:
        array = np.array(load_patch(os.path.join(data_dir, name)))
        if divided_liver:
            array = np.asarray(array, np.float32)
            liver_density = liver_density_dict[name[:name.rfind('_')]]
            for i in range(len(phasenames)):
                array[:, :, i] = (1.0 * array[:, :, i]) / (1.0 * liver_density[i])
        array = array.flatten()
        if len(array) != patch_size * patch_size * 3:
            continue
        features.append(array)
    # features = [np.array(Image.open(os.path.join(data_dir, name))).flatten() for name in names]
    return np.array(features)


def generate_density_feature_multidir(data_dirs):
    features = []
    results = []
    pool = Pool()
    for data_dir in data_dirs:
        result = pool.apply_async(generate_density_feature, args=(data_dir, ))
        results.append(result)

    # pool.close()
    # pool.join()
    for index, data_dir in enumerate(data_dirs):
        features.extend(results[index].get())
        print 'extracting features end from ', data_dir
    print np.shape(features)
    return np.array(features)


def do_kmeans(fea, vocabulary_size=128):
    print 'fea shape is ', np.shape(fea), ' vocabulary size is ', vocabulary_size
    kmeans_obj = KMeans(n_clusters=vocabulary_size, n_jobs=8).fit(fea)
    cluster_centroid_objs = kmeans_obj.cluster_centers_
    # np.save(
    #     './cluster_centroid_objs_'+str(vocabulary_size) + '_' + str(divided_liver) + '.npy',
    #     cluster_centroid_objs
    # )
    print np.shape(cluster_centroid_objs)
    return cluster_centroid_objs
def extract_interior_patch_npy(dir_name, suffix_name, patch_size, patch_step=1, flatten=False):
    '''
    提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
    :param patch_step: 提取ｐａｔｃｈ的步长
    :param flatten: 是否展开
    :return: None
    '''
    count = 0
    names = os.listdir(dir_name)
    patches_arr = []
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            flag = True
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)

                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                    flag = False
                    continue
                mask_image = image_erode(mask_image, 5)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
                # show_image(mhd_image)
            if not flag:
                continue
            mask_images = convert2depthlaster(mask_images)
            mhd_images = convert2depthlaster(mhd_images)
            count += 1
            [width, height, depth] = list(np.shape(mhd_images))
            patch_count = 1
            # if width * height >= 900:
            #     patch_step = int(math.sqrt(width * height / 100))
            patches = []
            for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                    cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                               j - patch_size / 2: j + patch_size / 2, :]) / (
                                    (patch_size - 1) * (patch_size - 1) * 3)) < 0.9:
                        continue
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                    # print save_path
                    # np.save(save_path, np.array(cur_patch))
                    # patches.append(cur_patch)
                    patch_count += 1
                    if flatten:
                        patches.append(np.array(cur_patch).flatten())
                    else:
                        patches.append(cur_patch)
            if patch_count == 1:
                continue
                # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                # roi_image.save(save_path)
            patches_arr.append(patches)
    return np.array(patches_arr)


def extract_boundary_patch_npy(dir_name, suffix_name, patch_size, patch_step=1, flatten=False):
    '''
    提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
    :param patch_step: 提取ｐａｔｃｈ的步长
    :param flatten: 是否展开
    :return: None
    '''
    count = 0
    names = os.listdir(dir_name)
    patches_arr = []
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            flag = True
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)

                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)

                if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                    flag = False
                    continue
                interior_boundary = image_erode(mask_image, 5)
                expand_boundary = image_expand(mask_image, 10)
                mask_image = np.asarray(np.logical_and(interior_boundary==0, expand_boundary==1), np.uint8)
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
                # show_image(mhd_image)
            if not flag:
                continue
            mask_images = convert2depthlaster(mask_images)
            mhd_images = convert2depthlaster(mhd_images)
            count += 1
            [width, height, depth] = list(np.shape(mhd_images))
            patch_count = 1
            # if width * height >= 900:
            #     patch_step = int(math.sqrt(width * height / 100))
            patches = []
            for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                    cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                               j - patch_size / 2: j + patch_size / 2, :]) / (
                                    (patch_size - 1) * (patch_size - 1) * 3)) < 0.7:
                        continue
                    if flatten:
                        patches.append(np.array(cur_patch).flatten())
                    else:
                        patches.append(cur_patch)
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                    # print save_path
                    # np.save(save_path, np.array(cur_patch))
                    patch_count += 1
            if patch_count == 1:
                continue
                # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                # roi_image.save(save_path)
            patches_arr.append(patches)
    print count
    return patches_arr

def extract_interior_boundary_patch_npy(dir_name, suffix_name, patch_size, patch_step=1, flatten=False):
    '''
    提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
    :param patch_step: 提取ｐａｔｃｈ的步长
    :param flatten: 是否展开
    :return: None
    '''
    count = 0
    names = os.listdir(dir_name)
    boundary_patches_arr = []
    interior_patches_arr = []
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            boundary_mask_images = []
            boundary_mhd_images = []
            interior_mask_images = []
            interior_mhd_images = []
            boundary_flag = True
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)

                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)

                if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                    boundary_flag = False
                    continue
                interior_boundary = image_erode(mask_image, 5)
                expand_boundary = image_expand(mask_image, 10)
                boundary_mask_image = np.asarray(np.logical_and(interior_boundary==0, expand_boundary==1), np.uint8)
                boundary_mask_image = boundary_mask_image[xmin: xmax, ymin: ymax]
                boundary_mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                # boundary_mhd_image[boundary_mask_image != 1] = 0
                boundary_mask_images.append(boundary_mask_image)
                boundary_mhd_images.append(boundary_mhd_image)


                [xmin, xmax, ymin, ymax] = get_boundingbox(interior_boundary)
                interior_mask_image = interior_boundary[xmin: xmax, ymin: ymax]
                interior_mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                # interior_mhd_image[mask_image != 1] = 0
                interior_mask_images.append(interior_mask_image)
                interior_mhd_images.append(interior_mhd_image)
                # show_image(mhd_image)
            if not boundary_flag:
                continue
            boundary_mask_images = convert2depthlaster(boundary_mask_images)
            boundary_mhd_images = convert2depthlaster(boundary_mhd_images)
            interior_mask_images = convert2depthlaster(interior_mask_images)
            interior_mhd_images = convert2depthlaster(interior_mhd_images)
            count += 1
            [boundary_width, boundary_height, _] = list(np.shape(boundary_mhd_images))
            [interior_width, interior_height, _] = list(np.shape(interior_mhd_images))
            boundary_patch_count = 1
            boundary_patches = []

            interior_patch_count = 1
            interior_patches = []
            for i in range(patch_size / 2, boundary_width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, boundary_height - patch_size / 2, patch_step):
                    cur_patch = boundary_mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(boundary_mask_images[i - patch_size / 2:i + patch_size / 2,
                               j - patch_size / 2: j + patch_size / 2, :]) / (
                                    (patch_size - 1) * (patch_size - 1) * 3)) < 0.7:
                        continue
                    if flatten:
                        boundary_patches.append(np.array(cur_patch).flatten())
                    else:
                        boundary_patches.append(cur_patch)
                    boundary_patch_count += 1
            for i in range(patch_size / 2, interior_width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, interior_height - patch_size / 2, patch_step):
                    cur_patch = interior_mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(interior_mask_images[i - patch_size / 2:i + patch_size / 2,
                               j - patch_size / 2: j + patch_size / 2, :]) / (
                                    (patch_size - 1) * (patch_size - 1) * 3)) < 0.9:
                        continue
                    if flatten:
                        interior_patches.append(np.array(cur_patch).flatten())
                    else:
                        interior_patches.append(cur_patch)
                    interior_patch_count += 1
            if interior_patch_count == 1 or boundary_patch_count == 1:
                continue
            boundary_patches_arr.append(boundary_patches)
            interior_patches_arr.append(interior_patches)
    print count
    print np.shape(boundary_patches_arr)
    return boundary_patches_arr, interior_patches_arr

def extract_patch(dir_name, suffix_name, patch_size, patch_step=1, flatten=False):
    '''
    提取指定类型病灶的ｐａｔｃｈ
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param patch_step: 提取ｐａｔｃｈ的步长
    :return: patch_arr (图像的个数, patch的个数)
    '''
    count = 0
    names = os.listdir(dir_name)
    patches_arr = []
    paths = []
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            paths.append(os.path.join(dir_name, name))
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path, rejust=True)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
                # show_image(mhd_image)
            mask_images = convert2depthlaster(mask_images)
            mhd_images = convert2depthlaster(mhd_images)
            # show_image(mhd_images)
            count += 1
            [width, height, depth] = list(np.shape(mhd_images))
            patch_count = 1
            # if width * height >= 400:
            #     patch_step = int(math.sqrt(width * height / 100))
            patches = []
            for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                    cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(
                            mask_images[i - patch_size / 2:i + patch_size / 2, j - patch_size / 2: j + patch_size / 2,
                            :]) / ((patch_size - 1) * (patch_size - 1) * 3)) < 0.9:
                        continue
                    patch_count += 1
                    if flatten:
                        patches.append(np.array(cur_patch).flatten())
                    else:
                        patches.append(cur_patch)
            if patch_count == 1:
                continue
            patches_arr.append(patches)
    return np.array(patches_arr)

def generate_train_val_features(boundary_cluster_centroid_path, interior_cluster_centriod_path, extract_patch_function):
    train_boundary_patches = []
    train_interior_patches = []
    train_labels = []
    for i in range(category_number):
        boundary_patches, interior_patches = extract_patch_function(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/train',
                str(i),
                patch_size=(patch_size + 1),
                flatten=True
            )
        train_labels.extend([i] * len(boundary_patches))
        train_boundary_patches.extend(
            boundary_patches
        )
        train_interior_patches.extend(interior_patches)
    train_boundary_patches = np.array(train_boundary_patches)
    train_interior_patches = np.array(train_interior_patches)
    print 'train_boundary_patches shape is ', np.shape(train_boundary_patches)
    print 'train_interior_patches shape is ', np.shape(train_interior_patches)

    val_boundary_patches = []
    val_interior_patches = []
    val_labels = []
    for i in range(category_number):
        boundary_patches, interior_patches = extract_patch_function(
            '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/test',
            str(i),
            patch_size=(patch_size + 1),
            flatten=True
        )
        val_labels.extend([i] * len(boundary_patches))
        val_boundary_patches.extend(
            boundary_patches
        )
        val_interior_patches.extend(interior_patches)
    val_boundary_patches = np.array(val_boundary_patches)
    val_interior_patches = np.array(val_interior_patches)
    print 'train_boundary_patches shape is ', np.shape(val_boundary_patches)
    print 'train_interior_patches shape is ', np.shape(val_interior_patches)

    boundary_cluster_centroid_arr = np.load(boundary_cluster_centroid_path)
    interior_cluster_centroid_arr = np.load(interior_cluster_centriod_path)
    train_boundary_features = []
    train_interior_features = []
    for i in range(len(train_boundary_patches)):
        train_boundary_features.append(
            generate_patches_representer(train_boundary_patches[i], boundary_cluster_centroid_arr).squeeze()
        )
        train_interior_features.append(
            generate_patches_representer(train_interior_patches[i], interior_cluster_centroid_arr).squeeze()
        )
    val_boundary_features = []
    val_interior_features = []
    for i in range(len(val_boundary_patches)):
        val_boundary_features.append(
            generate_patches_representer(val_boundary_patches[i], boundary_cluster_centroid_arr).squeeze()
        )
        val_interior_features.append(
            generate_patches_representer(val_interior_patches[i], interior_cluster_centroid_arr).squeeze()
        )
    print 'the shape of train boundary features is ', np.shape(train_boundary_features)
    print 'the shape of train interior features is ', np.shape(train_interior_features)
    print 'the shape of val boundary features is ', np.shape(val_boundary_features)
    print 'the shape of val interior features is ', np.shape(val_interior_features)
    # scio.savemat(
    #     './data_128_False.mat',
    #     {
    #         'train_features': train_features,
    #         'train_labels': train_labels,
    #         'val_features': val_features,
    #         'val_labels': val_labels
    #     }
    # )
    return train_boundary_features, train_interior_features, train_labels, \
           val_boundary_features, val_interior_features, val_labels

def cal_distance(patches, center):
    '''

    :param patches: None 49
    :param center: 128 * 49
    :return:
    '''
    patches2 = np.multiply(patches, patches)
    center2 = np.multiply(center, center)
    patchdotcenter = np.array(np.dot(np.mat(patches), np.mat(center).T))  # None * 128
    patches2sum = np.sum(patches2, axis=1)  # None
    center2sum = np.sum(center2, axis=1)    # 128
    distance_arr = np.zeros([len(patches2sum), len(center2sum)])
    for i in range(len(patches2sum)):
        for j in range(len(center2sum)):
            distance_arr[i, j] = patches2sum[i] + center2sum[j] - 2 * patchdotcenter[i, j]
    return distance_arr

def generate_patches_representer(patches, cluster_centers):
    '''
    用词典表示一组patches
    :param patches: 表示一组patch　（None, 192）
    :param cluster_centers:　(vocabulary_size, 192)
    :return: (1, vocabulary_size) 行向量　表示这幅图像
    '''
    print np.shape(patches)
    print np.shape(cluster_centers)
    shape = list(np.shape(cluster_centers))
    mat_cluster_centers = np.mat(cluster_centers)
    mat_patches = np.mat(patches)
    mat_distance = cal_distance(mat_patches, mat_cluster_centers)# (None, vocabulary_size)
    represented_vector = np.zeros([1, shape[0]])
    for i in range(len(mat_distance)):
        distance_vector = np.array(mat_distance[i])
        min_index = np.argmin(distance_vector)
        represented_vector[0, min_index] += 1
    return represented_vector

def generate_dictionary(patch_dir, cluster_save_path, target_labels=[0, 1, 2, 3], cluster_num=256, pre_class_num=30000):
    from LeaningBased.selected_patches import return_patches_multidir
    features = return_patches_multidir(
        patch_dir,
        subclass_names=['train', 'test'],
        target_label=target_labels,
        pre_class_num=pre_class_num
    )
    vocabulary = do_kmeans(features, vocabulary_size=cluster_num)
    np.save(cluster_save_path, vocabulary)


def execute_classify(train_features, train_labels, test_features, test_labels):
    from LeaningBased.BoVW_DualDict.classification import SVM, LinearSVM, KNN
    predicted_label, c_params, g_params, target_c, target_g, accs = SVM.do(train_features, train_labels, test_features,
                                                                           test_labels,
                                                                           adjust_parameters=True)
    calculate_acc_error(predicted_label, test_labels)
    return accs
if __name__ == '__main__':
    iterator_num = 20
    for iterator_index in range(10, iterator_num):
        print 'iterator_index is ', iterator_index
        # 提取内部字典
        generate_dictionary(
            '/home/give/Documents/dataset/ICPR2018/dual-dict/interior',
            './interior_dictionary.npy',
            cluster_num=128,
            pre_class_num=30000,
        )
        # 提取边界字典
        generate_dictionary(
            '/home/give/Documents/dataset/ICPR2018/dual-dict/boundary',
            './boundary_dictionary.npy',
            cluster_num=32,
            pre_class_num=10000
        )

        # 构造特征
        train_boundary_features, train_interior_features, train_labels, \
        val_boundary_features, val_interior_features, val_labels = generate_train_val_features(
            './boundary_dictionary.npy',
            './interior_dictionary.npy',
            extract_interior_boundary_patch_npy)
        scio.savemat(
            './features.mat',
            {
                'train_features': np.concatenate([train_boundary_features, train_interior_features], axis=1),
                'train_labels': train_labels,
                'test_features': np.concatenate([val_boundary_features, val_interior_features], axis=1),
                'test_labels': val_labels
            }
        )

        # 丢进分类器训练
        data = scio.loadmat('./features.mat')
        train_features = data['train_features']
        train_labels = data['train_labels']
        test_features = data['test_features']
        test_labels = data['test_labels']
        accs = execute_classify(train_features, train_labels, test_features, test_labels)

        # 保存结果
        parent_dir = '/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW_DualDict/find_best'
        cur_dir = os.path.join(parent_dir, str(iterator_index) + '_' + str(max(accs)))
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        cur_path = os.path.join(cur_dir, 'features.mat')
        copy('./features.mat', cur_path)
        print 'iterator_index is ', iterator_index, '  accuracy is ', max(accs)

    # 丢进分类器训练
    # data = scio.loadmat('/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW_DualDict/find_best/13_0.68/features.mat')
    # train_features = data['train_features']
    # train_labels = data['train_labels']
    # test_features = data['test_features']
    # test_labels = data['test_labels']
    # accs = execute_classify(train_features, train_labels, test_features, test_labels)