import os
import numpy as np
from ExtractPatches import extract_patches_multidir
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


def generate_representor_version2(data_dir, dictionary_path, subclass):
    kmeans_model = joblib.load(dictionary_path)
    shape_vocabulary = np.shape(kmeans_model.cluster_centers_)
    vocabulary_size = shape_vocabulary[0]
    representers = []
    patches, coding_labeles, labeles = extract_patches_multidir(data_dir, subclasses=[subclass], return_flag=True)
    all_patches = []
    counts = []
    for case_index, cur_patches in enumerate(patches):
        print np.shape(cur_patches)
        all_patches.extend(cur_patches)
        counts.append(len(cur_patches))
    print 'all patches shape are ', np.shape(all_patches)
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


def generate_representor_multidir(data_dir, patch_dir, reload=None):
    if reload is None or (reload == False):
        train_features, train_labels = generate_representor_version2(data_dir, dictionary_path=patch_dir,
                                                             subclass='train')
        scio.savemat('./training.mat', {
            'features': train_features,
            'labels': train_labels
        })
        test_features, test_labels = generate_representor_version2(data_dir, dictionary_path=patch_dir,
                                                          subclass='test')
        scio.savemat('./testing.mat', {
            'features': test_features,
            'labels': test_labels
        })
        val_features, val_labels = generate_representor_version2(data_dir, dictionary_path=patch_dir, subclass='val')
        scio.savemat('./validation.mat', {
            'features': val_features,
            'labels': val_labels
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
        patch_dir='/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-CO/vocabulary.model',
        data_dir='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP',
        reload=None,
    )