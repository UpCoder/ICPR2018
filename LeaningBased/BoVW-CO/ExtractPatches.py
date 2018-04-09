from skimage.feature import local_binary_pattern
from utils.Tools import read_mhd_image, get_boundingbox, check_save_path, split_array, convert2depthlaster
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


def flatten_multiphase(arr):
    shape = list(np.shape(arr))
    res = []
    for i in range(shape[2]):
        res.extend(np.array(arr[:, :, i]).flatten())
    return res


def convert_coding(file_dir):
    pv_mask_image, pv_mhd_image = read_from_dir(file_dir)
    [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)
    roi_image = pv_mhd_image[x_min:x_max, y_min: y_max]
    after_conding = local_binary_pattern(roi_image, 8, 3, 'uniform')
    return after_conding


def extract_patches_multifiles(data_dir, names, target_label, patch_size, patch_step, save_dir):
    patches = []
    labeles = []
    coding_labeles = []
    for name in names:
        if name is not None and not name.endswith(target_label):
            continue

        cur_data_dir = os.path.join(data_dir, name)
        patches_save_path = os.path.join(cur_data_dir, 'patches.npy')
        coding_labeles_path = os.path.join(cur_data_dir, 'coding_labeles.npy')
        labeles_path = os.path.join(cur_data_dir, 'labeles.npy')
        mask_images = []
        mhd_images = []
        if not os.path.exists(patches_save_path):
            for phasename in ['NC', 'ART', 'PV']:
                image_path = glob(os.path.join(data_dir, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(data_dir, name, phasename + '_Registration.mhd')
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
            cur_patches = []
            cur_coding_labeles = []
            mask_images = convert2depthlaster(mask_images)
            coding_image = convert_coding(cur_data_dir)
            mhd_images = convert2depthlaster(mhd_images)
            [width, height, _] = list(np.shape(mhd_images))
            print 'extract patches from ', cur_data_dir, ' at ', str(
                os.getpid()), ' corresponding size is [', width, height, ']'
            for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                    cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2 + 1, j - patch_size / 2: j + patch_size / 2 + 1, :]
                    cur_shape = list(np.shape(cur_patch))
                    if cur_shape[0] != patch_size or cur_shape[1] != patch_size:
                        continue
                    cur_label = target_label
                    cur_coding_label = coding_image[i, j]
                    if save_dir is not None:
                        save_path_dir = os.path.join(save_dir, str(cur_coding_label), str(cur_label))
                        if os.path.exists(save_path_dir):
                            cur_id = len(os.listdir(save_path_dir))
                        else:
                            cur_id = 0
                        save_path = os.path.join(save_dir, str(cur_coding_label), str(cur_label), str(cur_id) + '.npy')
                        check_save_path(save_path)
                        np.save(save_path, cur_patch)
                    else:
                        cur_patches.append(flatten_multiphase(np.array(cur_patch)))
                        cur_coding_labeles.append(cur_coding_label)
            if save_dir is None:
                if len(cur_patches) == 0:
                    continue
                patches.append(cur_patches)
                coding_labeles.append(cur_coding_labeles)
                labeles.append(int(target_label))
            np.save(patches_save_path, cur_patches)
            np.save(coding_labeles_path, cur_coding_labeles)

        else:
            cur_patches = np.load(patches_save_path)
            cur_coding_labeles = np.load(coding_labeles_path)
            coding_labeles.append(cur_coding_labeles)
            patches.append(cur_patches)
            labeles.append(int(target_label))
    print len(patches), len(coding_labeles), len(labeles)
    return patches, coding_labeles, labeles


def extract_patches_singledir(data_dir, target_label, patch_size, patch_step, save_dir, multiprocess=8):
    names = os.listdir(data_dir)
    patches = []
    labeles = []
    coding_labeles = []
    if multiprocess is None:
        patches, coding_labeles, labeles = extract_patches_multifiles(data_dir, names, target_label, patch_size,
                                                                      patch_step, None)
    else:
        names_group = split_array(names, multiprocess)
        pool = Pool()
        results = []
        for i in range(multiprocess):
            result = pool.apply_async(extract_patches_multifiles,
                                      (data_dir, names_group[i], target_label, patch_size, patch_step, None,))
            results.append(result)
        pool.close()
        pool.join()
        for i in range(multiprocess):
            cur_patches, cur_coding_labeles, cur_labeles = results[i].get()
            patches.extend(cur_patches)
            labeles.extend(cur_labeles)
            coding_labeles.extend(cur_coding_labeles)

    return patches, coding_labeles, labeles


def extract_patches_multidir(data_dir, subclasses=['train', 'val', 'test'], target_labels=[0, 1, 2, 3],
                             patch_size=7, patch_step=1,
                             save_path='/home/give/Documents/dataset/ICPR2018/BoVW-CO/data.mat', return_flag=False):
    patches = []
    labeles = []
    coding_labeles = []
    for subclass in subclasses:
        for target_label in target_labels:
            cur_data_dir = os.path.join(data_dir, subclass)
            cur_patches, cur_coding_labels, cur_labeles = extract_patches_singledir(cur_data_dir, str(target_label),
                                                                                    patch_size=patch_size,
                                                                                    patch_step=patch_step,
                                                                                    save_dir=None, multiprocess=8)
            patches.extend(cur_patches)
            coding_labeles.extend(cur_coding_labels)
            labeles.extend(cur_labeles)
    if return_flag:
        return patches, coding_labeles, labeles
    if save_path is not None:
        scio.savemat(save_path, {
            'patches': patches,
            'labeles': labeles
        })
        save_dict = {}

        for i, cur_patches in enumerate(patches):
            cur_label = labeles[i]
            if str(cur_label) in save_dict.keys():
                save_dict[str(cur_label)].extend(cur_patches)
            else:
                save_dict[str(cur_label)] = []
        scio.savemat(save_path, save_dict)
        # save_path = os.path.join(save_dir, 'save_dict.mat')
        # check_save_path(save_path)
        # # scio.savemat(save_path, save_dict)
        # for subkey in save_dict.keys():
        #     for subsubkey in save_dict[subkey].keys():
        #         save_path = os.path.join(save_dir, str(subkey), str(subsubkey), 'data.npy')
        #         check_save_path(save_path)
        #         np.save(save_path, save_dict[subkey][subsubkey])

if __name__ == '__main__':
    extract_patches_multidir('/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP')
    # data = scio.loadmat('/home/give/Documents/dataset/ICPR2018/BoVW-CO/data.mat')
    # for key in data.keys():
    #     print np.shape(data[key])