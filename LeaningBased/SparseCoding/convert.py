import numpy as np
import scipy.io as scio
import os


def convertNPY2Mat(path, save_path):
    data_arr = []
    label_arr = []
    for typeid in range(4):
        cur_path = os.path.join(path, str(typeid), 'data.npy')
        cur_data = np.load(cur_path)
        cur_label = [typeid] * len(cur_data)
        data_arr.extend(cur_data)
        label_arr.extend(cur_label)
    scio.savemat(save_path, {
        'data': data_arr,
        'label': label_arr
    })

if __name__ == '__main__':
    convertNPY2Mat(
        '/home/give/Documents/dataset/ICPR2018/BoVW-TextSpecific/0.0',
        '/home/give/Documents/dataset/ICPR2018/BoVW-TextSpecific/0.0/patches.mat'
    )
    patches = scio.loadmat('/home/give/Documents/dataset/ICPR2018/BoVW-TextSpecific/0.0/patches.mat')
    print patches
    for key in patches.keys():
        print np.shape(patches[key])