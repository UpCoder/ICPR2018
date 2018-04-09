import os
import numpy as np
from ExtractPatches import extract_patches_multidir
from utils.Tools import calculate_acc_error
import scipy.io as scio
from sklearn.externals import joblib
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer

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


def convert2str(patches_predicted, counts):
    start = 0
    res_strs = []
    for cur_index, cur_count in enumerate(counts):
        cur_str = ' '.join([str(element) for element in patches_predicted[start: start + cur_count]])
        start += cur_count
        res_strs.append(cur_str)
    return res_strs


def generate_representor_version2(data_dir, dictionary_path, subclass, vectorizer=None):
    save_dir = '/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-NGram/patches'
    save_dir = os.path.join(save_dir, subclass)
    paths = glob(os.path.join(save_dir, '*.npy'))

    kmeans_model = joblib.load(dictionary_path)
    shape_vocabulary = np.shape(kmeans_model.cluster_centers_)
    vocabulary_size = shape_vocabulary[0]
    if len(paths) == 0:
        patches, _, labeles = extract_patches_multidir(data_dir, subclasses=[subclass], return_flag=True,
                                                                    patch_size=3)
    else:
        patches = []
        labeles = []
        for path in paths:
            patches.append(np.load(path))
            labeles.append(int(os.path.basename(path).split('.npy')[0].split('_')[1]))
    all_patches = []
    counts = []
    for case_index, cur_patches in enumerate(patches):
        # print np.shape(cur_patches)
        if len(paths) == 0:
            np.save(os.path.join(save_dir, str(case_index) + '_' + str(labeles[case_index])), cur_patches)

        all_patches.extend(cur_patches)
        counts.append(len(cur_patches))
    print 'all patches shape are ', np.shape(all_patches)
    predicted_labels = kmeans_model.predict(all_patches)
    start = 0
    strs = convert2str(predicted_labels, counts)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer='char', min_df=1, ngram_range=(1, 3), use_idf=False, stop_words=None)
        vectorizer = vectorizer.fit(strs)
    crs_matrix = vectorizer.transform(strs).toarray()
    return np.asarray(crs_matrix, np.float32), labeles, vectorizer


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
    if reload is None or (not reload):
        train_features, train_labels, vectorizer = generate_representor_version2(data_dir, dictionary_path=patch_dir,
                                                             subclass='train')
        scio.savemat('./training.mat', {
            'features': train_features,
            'labels': train_labels
        })
        test_features, test_labels, _ = generate_representor_version2(data_dir, dictionary_path=patch_dir,
                                                                      vectorizer=vectorizer,
                                                                      subclass='test')
        scio.savemat('./testing.mat', {
            'features': test_features,
            'labels': test_labels
        })
        val_features, val_labels, _ = generate_representor_version2(data_dir, dictionary_path=patch_dir, subclass='val',
                                                                    vectorizer=vectorizer)
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
    print 'trian_features shape are: ', np.shape(train_features)
    print 'val_features shape are: ', np.shape(val_features)
    print 'test_features shape are: ', np.shape(test_features)
    print 'train_labels shape are: ', np.shape(train_labels)
    print 'val_labels shape are: ', np.shape(val_labels)
    print 'test_labels shape are: ', np.shape(test_labels)
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
        patch_dir='/home/give/PycharmProjects/ICPR2018/LeaningBased/BoVW-NGram/vocabulary.model',
        data_dir='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP',
        reload=None,
    )