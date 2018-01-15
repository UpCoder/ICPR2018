# -*- coding=utf-8 -*-
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
import scipy.io as scio


def get_probas_Frid_Adar():
    train_data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patch_ROI_Frid-Adar/train.npy.mat')
    train_features = train_data['features']
    train_labels = train_data['labels']
    train_features = np.squeeze(train_features)
    train_labels = np.squeeze(train_labels)

    test_data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patch_ROI_Frid-Adar/test.npy.mat')
    test_features = test_data['features']
    test_labels = test_data['labels']
    test_features = np.squeeze(test_features)
    test_labels = np.squeeze(test_labels)

    clf = SVC(C=1, gamma='auto', probability=True)
    clf.fit(train_features, train_labels)
    probas_ = clf.predict_proba(test_features)
    val_labels = label_binarize(test_labels, [0, 1, 2, 3])
    return probas_, val_labels


def get_probas_Patch_ROI():
    train_data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patch_ROI/train.npy.mat')
    train_features = train_data['features']
    train_labels = train_data['labels']
    train_features = np.squeeze(train_features)
    train_labels = np.squeeze(train_labels)

    test_data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patch_ROI/test.npy.mat')
    test_features = test_data['features']
    test_labels = test_data['labels']
    test_features = np.squeeze(test_features)
    test_labels = np.squeeze(test_labels)

    clf = SVC(C=1, gamma='auto', probability=True)
    clf.fit(train_features, train_labels)
    probas_ = clf.predict_proba(test_features)
    val_labels = label_binarize(test_labels, [0, 1, 2, 3])
    return probas_, val_labels


def get_probas_Patch():
    train_data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patched/train.npy.mat')
    train_features = train_data['features']
    train_labels = train_data['labels']
    train_features = np.squeeze(train_features)
    train_labels = np.squeeze(train_labels)

    test_data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/Net/ICIP/4-class/Patched/test.npy.mat')
    test_features = test_data['features']
    test_labels = test_data['labels']
    test_features = np.squeeze(test_features)
    test_labels = np.squeeze(test_labels)

    clf = SVC(C=1, gamma='auto', probability=True)
    clf.fit(train_features, train_labels)
    probas_ = clf.predict_proba(test_features)
    val_labels = label_binarize(test_labels, [0, 1, 2, 3])
    return probas_, val_labels


def plot_roc(labels, probas, class_num, names, colors):
    '''
    对比多组结果的roc曲线
    :param labels: label onehot编码格式，可以通过label_binarize方法得到　[group_num, sample_num, class_num]
    :param probas:[group_num, sample_num, class_num]
    :param names:　不同group的名字
    :param class_num:
    :return:
    '''
    plt.figure()
    for group_id in range(len(labels)):
        label = labels[group_id]
        probas_ = probas[group_id]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(class_num):
            fpr[i], tpr[i], _ = roc_curve(label[:, i], probas_[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        label = np.array(label)
        probas_ = np.array(probas_)
        flatten_label = label.flatten()
        flatten_probas_ = probas_.flatten()
        fpr["micro"], tpr["micro"], _ = roc_curve(flatten_label, flatten_probas_)
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])


        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(class_num):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= class_num
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
        plt.plot(fpr['micro'], tpr['micro'], label=(names[group_id] + ' micro-average ROC curve (area={0:0.2f})').format(roc_auc['micro']), color=colors[group_id][0], linestyle=':', linewidth=4)
        plt.plot(fpr['macro'], tpr['macro'], label=(names[group_id] + ' macro-average ROC curve (area={0:0.2f})').format(roc_auc['macro']), color=colors[group_id][1], linestyle=':', linewidth=4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Different Method')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    # probas_BoVW, label_BoVW = get_probas_RAWHeating()
    # probas_OurMethod, label_OurMethod = get_probas_OurMethod()
    # plot_roc([label_BoVW, label_OurMethod], [probas_BoVW, probas_OurMethod], names=['Heating map', 'THMG-SP'],
    #          colors=[[[1, 0, 1], [0, 0, 0]], [[0, 0, 1], [0, 1, 1]]], class_num=4)
    probas_Fird, label_Frid = get_probas_Frid_Adar()
    probas_patched, label_patched = get_probas_Patch()
    probas_PatchROI, label_PatchROI = get_probas_Patch_ROI()
    plot_roc([label_patched, label_PatchROI], [probas_patched, probas_PatchROI], names=['Patched\'s method', 'Our Method'],
              colors=[[[1, 0, 1], [0, 0, 0]], [[0, 0, 1], [0, 1, 1]]], class_num=4)