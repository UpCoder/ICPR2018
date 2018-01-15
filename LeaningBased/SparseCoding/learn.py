from ksvd import KSVD
import numpy.random as rn
from numpy import array, zeros, dot
import numpy as np
import scipy.io as scio

if __name__ == "__main__":
    data_path = '/home/give/Documents/dataset/ICPR2018/BoVW-TextSpecific/0.0/patches.mat'
    data = scio.loadmat(data_path)
    dict_size = 1024
    target_sparsity = 8
    n_examples = 153000
    dimension = 512
    print data.keys()
    X = data['data']
    print np.shape(X)

    result = KSVD(X, dict_size, target_sparsity, 50,
                  print_interval=25,
                  enable_printing=True, enable_threading=True)
    # print result