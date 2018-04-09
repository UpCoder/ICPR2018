import os
from LeaningBased.BoVW.ExtractPatches import read_from_dir, convert_coding
from utils.Tools import get_boundingbox, check_save_path
import numpy as np


def extract_patches(data_dir, name, target_label, patch_size, patch_step=1, save_dir=None):
    cur_data_dir = os.path.join(data_dir, name)
    print 'extract patches from ', cur_data_dir, ' at ', str(os.getpid())
    pv_mask_image, pv_mhd_image = read_from_dir(cur_data_dir)
    coding_image = convert_coding(cur_data_dir)
    [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)
    r = patch_size / 2
    cur_patches = []
    cur_coding_labeles = []
    for i in range(x_min, x_max, patch_step):
        for j in range(y_min, y_max, patch_step):
            cur_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
            cur_mask_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
            if ((1.0 * np.sum(cur_mask_patch)) / (1.0 * patch_size * patch_size)) < 0.1:
                continue
            cur_label = target_label
            cur_coding_label = coding_image[i - x_min, j - y_min]

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
                cur_patches.append(np.array(cur_patch).flatten())
                cur_coding_labeles.append(cur_coding_label)
    return cur_patches


def load_vocabulary(data_dir):
    return np.load(data_dir)
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

def generate_representer(dictionary_path, patches):
    dictionary = load_vocabulary(dictionary_path)
    shape = list(np.shape(dictionary))
    distance_arr = cal_distance(patches, dictionary)
    cur_case_representor = np.zeros([1, shape[0]])
    for i in range(len(distance_arr)):
        min_index = np.argmin(distance_arr[i])
        cur_case_representor[0, min_index] += 1
    return np.squeeze(np.array(cur_case_representor))


def do_svm(patches, labels, representer):
    print np.shape(patches), np.shape(labels)
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(patches, labels)
    return svc.predict(representer)
def load_training(data_path):
    import scipy.io as scio
    data = scio.loadmat(data_path)

    return data['features'], np.squeeze(data['labels'])

if __name__ == '__main__':
    import time
    start_time = time.time()
    data_path = '/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW/training.mat'
    training_features, training_labels = load_training(data_path)
    data_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/visualize'
    data_name = '3608006_3201151_0_0_2'
    dictionary_path = '/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW/dictionary.npy'
    patches = extract_patches(data_dir, data_name, 2, 7, patch_step=1, save_dir=None)
    representer = generate_representer(dictionary_path, patches)
    print np.shape(patches)
    do_svm(training_features, training_labels, representer)
    end_time = time.time()
    print 'cost time is ', (end_time - start_time)