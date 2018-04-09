import os
import numpy as np
from ExtractPatches import extract_patches_multidir, extract_patches_singledir, extract_patches_multifiles_boundary, extract_patches_multifiles_interior
from utils.Tools import calculate_acc_error
import scipy.io as scio
from sklearn.externals import joblib


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

def load_vocabulary(data_dir):
    return np.load(data_dir)


def generate_representor_version2(data_dir, dictionary_path, subclass, extract_patches_multifiles_function):
    kmeans_model = joblib.load(dictionary_path)
    shape_vocabulary = np.shape(kmeans_model.cluster_centers_)
    vocabulary_size = shape_vocabulary[0]
    representers = []

    cur_data_dir = os.path.join(data_dir, subclass)
    patches = []
    labeles = []
    for target_label in range(4):
        cur_patches, cur_labeles = extract_patches_singledir(cur_data_dir, str(target_label),
                                                             patch_size=7,
                                                             patch_step=1,
                                                             save_dir=None, multiprocess=8,
                                                             extract_patches_multifiles_function=extract_patches_multifiles_function)

        print np.shape(cur_patches), extract_patches_multifiles_function
        patches.extend(cur_patches)
        labeles.extend(cur_labeles)
    all_patches = []
    counts = []
    for case_index, cur_patches in enumerate(patches):
        print np.shape(cur_patches)
        all_patches.extend(cur_patches)
        counts.append(len(cur_patches))
    predicted_labels = kmeans_model.predict(all_patches)
    start = 0
    for case_index, count in enumerate(counts):
        cur_predicted_label = predicted_labels[start: start + count]
        representer = np.histogram(cur_predicted_label, bins=vocabulary_size, normed=True)[0]
        representers.append(np.array(representer).squeeze())
        start += count
    return representers, labeles


def generate_representor(data_dir, dictionary_path, subclass):
    dictionary = load_vocabulary(dictionary_path)
    shape_vocabulary = np.shape(dictionary)
    vocabulary_size = shape_vocabulary[0]
    representers = []
    patches, coding_labeles, labeles = extract_patches_multidir(data_dir, subclasses=[subclass], return_flag=True)
    all_patches = []
    counts = []
    for case_index, cur_patches in enumerate(patches):
        print np.shape(cur_patches)
        all_patches.extend(cur_patches)
        counts.append(len(cur_patches))
    all_distance_arr = cal_distance(all_patches, dictionary)
    start = 0
    for case_index, count in enumerate(counts):
        distance_arr = all_distance_arr[start: start + count]
        cur_case_representor = np.zeros([1, vocabulary_size])
        for i in range(len(distance_arr)):
            min_index = np.argmin(distance_arr[i])
            cur_case_representor[0, min_index] += 1
        representers.append(cur_case_representor.squeeze())
        start += count
        # patches_coding_labeles = {}
        # for patch_index, cur_patch in enumerate(cur_patches):
        #     cur_coding_label = coding_labeles[case_index][patch_index]
        #     if cur_coding_label not in patches_coding_labeles.keys():
        #         patches_coding_labeles[cur_coding_label] = []
        #     patches_coding_labeles[cur_coding_label].append(cur_patch)
        # for key in patches_coding_labeles.keys():
        #     cur_patches_coding_label = patches_coding_labeles[key]
        #     cur_vocabulary = vocabulary_dict[key]
        #     distance_arr = cal_distance(cur_patches_coding_label, cur_vocabulary)
        #     for i in range(len(distance_arr)):
        #         min_index = np.argmin(distance_arr[i])
        #         cur_case_representor[int(key), min_index] += 1
        # representers.append(cur_case_representor.flatten())
    return representers, labeles


def execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    from LeaningBased.BoVW_DualDict.classification import SVM, LinearSVM, KNN
    predicted_label, c_params, g_params, max_c, max_g, accs = SVM.do(train_features, train_labels, val_features,
                                                                     val_labels,
                                                                     adjust_parameters=True)
    predicted_label, accs = SVM.do(train_features, train_labels, test_features, test_labels,
                                   adjust_parameters=False, C=max_c, gamma=max_g)
    calculate_acc_error(predicted_label, test_labels)
    print 'ACA is ', accs
    return accs


def generate_representor_multidir(interior_path, boundary_path, patch_dir, reload=None):
    if reload is None:
        train_features_interior, train_labels_interior = generate_representor_version2(patch_dir,
                                                                                       dictionary_path=interior_path,
                                                                                       subclass='train',
                                                                                       extract_patches_multifiles_function=extract_patches_multifiles_interior)
        train_features_boundary, train_labels_boundary = generate_representor_version2(patch_dir,
                                                                                       dictionary_path=boundary_path,
                                                                                       subclass='train',
                                                                                       extract_patches_multifiles_function=extract_patches_multifiles_boundary)

        train_features = np.concatenate([train_features_interior, train_features_boundary], axis=1)
        train_labels = train_labels_interior
        scio.savemat('./training.mat', {
            'features': train_features,
            'labels': train_labels_interior
        })

        test_features_interior, test_labels_interior = generate_representor_version2(patch_dir,
                                                                                     dictionary_path=interior_path,
                                                                                     subclass='test',
                                                                                     extract_patches_multifiles_function=extract_patches_multifiles_interior)
        test_features_boundary, test_labels_boundary = generate_representor_version2(patch_dir,
                                                                                     dictionary_path=boundary_path,
                                                                                     subclass='test',
                                                                                     extract_patches_multifiles_function=extract_patches_multifiles_boundary)
        test_features = np.concatenate([test_features_interior, test_features_boundary], axis=1)
        test_labels = test_labels_interior
        scio.savemat('./testing.mat', {
            'features': test_features,
            'labels': test_labels_interior
        })

        val_features_interior, val_labels_interior = generate_representor_version2(patch_dir,
                                                                                   dictionary_path=interior_path,
                                                                                   subclass='val',
                                                                                   extract_patches_multifiles_function=extract_patches_multifiles_interior)
        val_features_boundary, val_labels_boundary = generate_representor_version2(patch_dir,
                                                                                   dictionary_path=boundary_path,
                                                                                   subclass='val',
                                                                                   extract_patches_multifiles_function=extract_patches_multifiles_boundary)
        val_features = np.concatenate([val_features_interior, val_features_boundary], axis=1)
        val_labels = val_labels_interior
        scio.savemat('./validation.mat', {
            'features': val_features,
            'labels': val_labels_boundary
        })
    else:
        train_data = scio.loadmat('./training.mat')
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = scio.loadmat('./testing.mat')
        test_features = test_data['features']
        test_labels = test_data['labels']

        val_data = scio.loadmat('./validation.mat')
        val_features = val_data['features']
        val_labels = val_data['labels']
    acc = execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels)
    return acc

if __name__ == '__main__':
    # from learn import generate_dictionary
    # for i in range(10):
    #     generate_dictionary(
    #         save_path='/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-CO/vocabulary/' + 'vocabulary_' + str(
    #             i) + '.npy', vocabulary_size=128)
    #     acc = generate_representor_multidir(
    #         patch_dir='/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-CO/vocabulary/'+'vocabulary_'+str(i)+'.npy',
    #         data_dir='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP')
    #     print 'Iterator %d, accuracy %.2f' % (i, acc)
    acc = generate_representor_multidir(
        interior_path='/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-DualDict-New/vocabulary_interior.model',
        boundary_path='/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-DualDict-New/vocabulary_boundary.model',
        patch_dir='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP',
        reload=True,
    )