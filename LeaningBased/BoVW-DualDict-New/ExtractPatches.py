from skimage.feature import local_binary_pattern
from utils.Tools import read_mhd_image, get_boundingbox, split_array, image_erode, image_expand
import os
from glob import glob
import numpy as np
from multiprocessing import Pool
import scipy.io as scio


def read_from_dir(data_dir, phasename='PV'):
    mhd_path = glob(os.path.join(data_dir, phasename + '_Image*.mhd'))[0]
    mhd_image = read_mhd_image(mhd_path)
    mask_path = os.path.join(data_dir, phasename + '_Registration.mhd')
    mask_image = read_mhd_image(mask_path)
    mhd_image = np.squeeze(mhd_image)
    mask_image = np.squeeze(mask_image)
    return mask_image, mhd_image


def convert_coding(file_dir):
    pv_mask_image, pv_mhd_image = read_from_dir(file_dir)
    [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)
    roi_image = pv_mhd_image[x_min:x_max, y_min: y_max]
    after_conding = local_binary_pattern(roi_image, 8, 3, 'uniform')
    return after_conding


def extract_patches_multifiles_interior(data_dir, names, target_label, patch_size, patch_step, save_dir):
    patches = []
    labeles = []
    for name in names:
        if name is not None and not name.endswith(target_label):
            continue
        cur_data_dir = os.path.join(data_dir, name)
        print 'extract patches from ', cur_data_dir, ' at ', str(os.getpid())
        pv_mask_image, pv_mhd_image = read_from_dir(cur_data_dir)
        pv_mask_image = image_erode(pv_mask_image, size=5)
        if np.sum(pv_mask_image == 1) < 30:
            continue
        [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)

        r = patch_size / 2
        cur_patches = []
        for i in range(x_min, x_max, patch_step):
            for j in range(y_min, y_max, patch_step):
                cur_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
                cur_mask_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
                if ((1.0 * np.sum(cur_mask_patch)) / (1.0 * patch_size * patch_size)) < 0.1:
                    continue

                cur_patches.append(np.array(cur_patch).flatten())
        if save_dir is None:
            # if len(cur_patches) == 0:
            #     continue
            patches.append(cur_patches)
            labeles.append(int(target_label))
    print len(patches), len(labeles)
    return patches, labeles


def extract_patches_multifiles_boundary(data_dir, names, target_label, patch_size, patch_step, save_dir):
    patches = []
    labeles = []
    for name in names:
        if name is not None and not name.endswith(target_label):
            continue
        cur_data_dir = os.path.join(data_dir, name)
        print 'extract patches from ', cur_data_dir, ' at ', str(os.getpid())
        pv_mask_image, pv_mhd_image = read_from_dir(cur_data_dir)
        erode_mask_image = image_erode(pv_mask_image, size=5)
        if np.sum(erode_mask_image == 1) < 30:
            continue
        expand_mask_image = image_expand(pv_mask_image, size=10)
        pv_mask_image = np.asarray(np.logical_and(erode_mask_image == 0, expand_mask_image == 1), np.uint8)
        [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)
        r = patch_size / 2
        cur_patches = []
        for i in range(x_min, x_max, patch_step):
            for j in range(y_min, y_max, patch_step):
                cur_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
                cur_mask_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
                if ((1.0 * np.sum(cur_mask_patch)) / (1.0 * patch_size * patch_size)) < 0.1:
                    continue

                cur_patches.append(np.array(cur_patch).flatten())
        if save_dir is None:
            # if len(cur_patches) == 0:
            #     continue
            patches.append(cur_patches)
            labeles.append(int(target_label))
    print len(patches), len(labeles)
    return patches, labeles


def extract_patches_singledir(data_dir, target_label, patch_size, patch_step, save_dir, multiprocess=8,
                              extract_patches_multifiles_function=extract_patches_multifiles_interior):
    names = os.listdir(data_dir)
    patches = []
    labeles = []
    if multiprocess is None:
        patches, labeles = extract_patches_multifiles_function(data_dir, names, target_label, patch_size,
                                                      patch_step, None)
    else:
        names_group = split_array(names, multiprocess)
        pool = Pool()
        results = []
        for i in range(multiprocess):
            result = pool.apply_async(extract_patches_multifiles_function,
                                      (data_dir, names_group[i], target_label, patch_size, patch_step, None,))
            results.append(result)
        pool.close()
        pool.join()
        for i in range(multiprocess):
            try:
                cur_patches, cur_labeles = results[i].get()
                patches.extend(cur_patches)
                labeles.extend(cur_labeles)
            except ValueError:
                pass
    return patches, labeles


def extract_patches_multidir(data_dir, subclasses=['train', 'val', 'test'], target_labels=[0, 1, 2, 3],
                             patch_size=7, patch_step=1,
                             save_dir='/home/give/Documents/dataset/ICPR2018/BoVW-DualDict/', return_flag=False):
    interior_patches = []
    interior_labeles = []
    boundary_patches = []
    boundary_labeles = []
    for subclass in subclasses:
        for target_label in target_labels:
            cur_data_dir = os.path.join(data_dir, subclass)
            cur_patches, cur_labeles = extract_patches_singledir(cur_data_dir, str(target_label),
                                                                 patch_size=patch_size,
                                                                 patch_step=patch_step,
                                                                 save_dir=None, multiprocess=8,
                                                                 extract_patches_multifiles_function=extract_patches_multifiles_interior)
            interior_patches.extend(cur_patches)
            interior_labeles.extend(cur_labeles)

            cur_patches, cur_labeles = extract_patches_singledir(cur_data_dir, str(target_label),
                                                                 patch_size=patch_size,
                                                                 patch_step=patch_step,
                                                                 save_dir=None, multiprocess=8,
                                                                 extract_patches_multifiles_function=extract_patches_multifiles_boundary)

            boundary_patches.extend(cur_patches)
            boundary_labeles.extend(cur_labeles)
    if return_flag:
        return interior_patches, interior_labeles, boundary_patches, boundary_labeles
    if save_dir is not None:
        scio.savemat(os.path.join(save_dir, 'interior_data.mat'), {
            'patches': interior_patches,
            'labeles': interior_labeles
        })
        save_dict = {}

        for i, cur_patches in enumerate(interior_patches):
            cur_label = interior_labeles[i]
            if str(cur_label) in save_dict.keys():
                save_dict[str(cur_label)].extend(cur_patches)
            else:
                save_dict[str(cur_label)] = []
        scio.savemat(os.path.join(save_dir, 'interior_data.mat'), save_dict)

        save_dict = {}
        for i, cur_patches in enumerate(boundary_patches):
            cur_label = boundary_labeles[i]
            if str(cur_label) in save_dict.keys():
                save_dict[str(cur_label)].extend(cur_patches)
            else:
                save_dict[str(cur_label)] = []
        scio.savemat(os.path.join(save_dir, 'boundary_data.mat'), save_dict)

if __name__ == '__main__':
    # extract_patches_multidir('/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP')
    data = scio.loadmat('/home/give/Documents/dataset/ICPR2018/BoVW-DualDict/boundary_data.mat')
    for key in data.keys():
        print key, ': ', np.shape(data[key])