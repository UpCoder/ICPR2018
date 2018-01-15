import numpy as np
import os
from PIL import Image

def visualize_centroid():
    data_path = '/home/give/PycharmProjects/MedicalImage/BoVW-Idit/interior_dictionary.npy'
    data = np.load(data_path)
    split_num = 64
    for i in range(len(data)):
        will_split = np.squeeze(data[i, :])
        image = np.zeros([8, 8, 3])
        for j in range(3):
            image[:, :, j] = np.reshape(will_split[j*split_num: (j+1)*split_num], [8, 8])
        print np.shape(image)
        img = Image.fromarray(np.asarray(image*255, np.uint8))
        img = img.resize([7, 7])
        img.save(os.path.join('/home/give/PycharmProjects/MedicalImage/BoVW-Idit/visualize/7*7', str(i) + '.jpg'))

if __name__ == '__main__':
    visualize_centroid()