import os
patches_num = 40000
from Tools import shuffle_array
import shutil

def selected_patches(source_dir, target_dir, num_limited):
    names = os.listdir(source_dir)
    names = shuffle_array(names)
    names = names[:num_limited]
    paths = [os.path.join(source_dir, name) for name in names]
    for path in paths:
        shutil.copy(path, os.path.join(target_dir, os.path.basename(path)))

def selected_patches_multidir():
    data_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/Frid-Adar'
    target_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/Frid-Adar/balance'
    subclass_patch_num = [50000, 30000, 50000]
    for index, subclass in enumerate(['train', 'val', 'test']):
        for typeid in [0, 1, 2, 3, 4]:
            selected_patches(
                os.path.join(data_dir, subclass, str(typeid)),
                os.path.join(target_dir, subclass, str(typeid)),
                subclass_patch_num[index]
            )

if __name__ == '__main__':
    selected_patches_multidir()