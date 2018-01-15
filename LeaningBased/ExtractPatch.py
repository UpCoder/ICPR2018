# -*- coding=utf-8 -*-
import os
from utils.Tools import read_mhd_image, get_boundingbox, convert2depthlaster, image_expand, image_erode
import numpy as np
from glob import glob
from PIL import Image
import scipy.io as scio
import math
phasenames = ['NC', 'ART', 'PV']

class ExtractPatch:
    @staticmethod
    def extract_patch(dir_name, suffix_name, save_dir, patch_size, patch_step=1):
        '''
        提取指定类型病灶的ｐａｔｃｈ
        :param patch_size: 提取ｐａｔｃｈ的大小
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :param patch_step: 提取ｐａｔｃｈ的步长
        :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
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
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                if width*height >= 400:
                    patch_step = int(math.sqrt(width*height/100))
                for i in range(patch_size/2, width - patch_size/2, patch_step):
                    for j in range(patch_size/2, height - patch_size/2, patch_step):
                        cur_patch = mhd_images[i-patch_size/2:i+patch_size/2, j-patch_size/2: j+patch_size/2, :]
                        if (np.sum(mask_images[i-patch_size/2:i+patch_size/2, j-patch_size/2: j+patch_size/2, :]) / ((patch_size-1) * (patch_size - 1) * 3)) < 0.9:
                            continue
                        save_path = os.path.join(save_dir, name+'_'+str(patch_count)+'.png')
                        patch_image = Image.fromarray(np.asarray(cur_patch, np.uint8))
                        patch_image.save(save_path)
                        patch_count += 1
                if patch_count == 1:
                    save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                    roi_image.save(save_path)
        print count

    @staticmethod
    def extract_patch_npy(dir_name, suffix_name, save_dir, patch_size, patch_step=1):
        '''
        提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
        :param patch_size: 提取ｐａｔｃｈ的大小
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :param patch_step: 提取ｐａｔｃｈ的步长
        :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path)
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
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                if width * height >= 900:
                    patch_step = int(math.sqrt(width * height / 160))
                for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                    for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                        cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                    j - patch_size / 2: j + patch_size / 2, :]
                        if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                                   j - patch_size / 2: j + patch_size / 2, :]) / (
                                (patch_size - 1) * (patch_size - 1) * 3)) < 0.9:
                            continue
                        save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        # print save_path
                        np.save(save_path, np.array(cur_patch))
                        patch_count += 1
                if patch_count == 1:
                    continue
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                    # roi_image.save(save_path)
        print count

    @staticmethod
    def extract_interior_patch_npy(dir_name, suffix_name, save_dir, patch_size, patch_step=1, erode_size=1):
        '''
        提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
        :param patch_size: 提取ｐａｔｃｈ的大小
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :param patch_step: 提取ｐａｔｃｈ的步长
        :param erode_size: 向内缩的距离，因为我们需要确定内部区域,所以为了得到内部区域，我们就将原区域向内缩以得到内部区域
        :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                flag = True
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path)
                    mhd_image = np.squeeze(mhd_image)
                    # show_image(mhd_image)
                    mask_image = read_mhd_image(mask_path)
                    mask_image = np.squeeze(mask_image)

                    [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                    if (xmax - xmin) <= erode_size or (ymax - ymin) <= erode_size:
                        flag = False
                        continue
                    mask_image = image_erode(mask_image, erode_size)
                    [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                    mask_image = mask_image[xmin: xmax, ymin: ymax]
                    mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                    # mhd_image[mask_image != 1] = 0
                    mask_images.append(mask_image)
                    mhd_images.append(mhd_image)
                    # show_image(mhd_image)
                if not flag:
                    continue
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                # if width * height >= 900:
                #     patch_step = int(math.sqrt(width * height / 100))
                for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                    for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                        cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                    j - patch_size / 2: j + patch_size / 2, :]
                        if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                                   j - patch_size / 2: j + patch_size / 2, :]) / (
                                        (patch_size - 1) * (patch_size - 1) * 3)) < 0.5:
                            continue
                        save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        # print save_path
                        np.save(save_path, np.array(cur_patch))
                        patch_count += 1
                if patch_count == 1:
                    continue
        print count

    @staticmethod
    def extract_boundary_patch_npy(dir_name, suffix_name, save_dir, patch_size, patch_step=1):
        '''
        提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
        :param patch_size: 提取ｐａｔｃｈ的大小
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :param patch_step: 提取ｐａｔｃｈ的步长
        :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                flag = True
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path)
                    mhd_image = np.squeeze(mhd_image)
                    # show_image(mhd_image)
                    mask_image = read_mhd_image(mask_path)
                    mask_image = np.squeeze(mask_image)

                    [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)

                    if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                        flag = False
                        continue
                    interior_boundary = image_erode(mask_image, 5)
                    expand_boundary = image_expand(mask_image, 10)
                    mask_image = np.asarray(np.logical_and(interior_boundary==0, expand_boundary==1), np.uint8)
                    mask_image = mask_image[xmin: xmax, ymin: ymax]
                    mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                    # mhd_image[mask_image != 1] = 0
                    mask_images.append(mask_image)
                    mhd_images.append(mhd_image)
                    # show_image(mhd_image)
                if not flag:
                    continue
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                # if width * height >= 900:
                #     patch_step = int(math.sqrt(width * height / 100))
                for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                    for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                        cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                    j - patch_size / 2: j + patch_size / 2, :]

                        if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                                   j - patch_size / 2: j + patch_size / 2, :]) / (
                                        (patch_size - 1) * (patch_size - 1) * 3)) < 0.1:
                            continue

                        save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        # print save_path
                        np.save(save_path, np.array(cur_patch))
                        patch_count += 1
                if patch_count == 1:
                    continue
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                    # roi_image.save(save_path)
        print count

    @staticmethod
    def extract_interior_boundary_patch_npy(dir_name, suffix_name, save_dir, patch_size, patch_step):
        '''
            提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
            :param patch_size: 提取ｐａｔｃｈ的大小
            :param dir_name: 目前所有病例的存储路径
            :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
            :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
            :param patch_step: 提取ｐａｔｃｈ的步长
            :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                flag = True
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path)
                    mhd_image = np.squeeze(mhd_image)
                    # show_image(mhd_image)
                    mask_image = read_mhd_image(mask_path)
                    mask_image = np.squeeze(mask_image)

                    [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)

                    if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                        flag = False
                        continue
                    interior_boundary = image_erode(mask_image, 5)
                    expand_boundary = image_expand(mask_image, 10)
                    mask_image = np.asarray(np.logical_and(interior_boundary == 0, expand_boundary == 1), np.uint8)
                    mask_image = mask_image[xmin: xmax, ymin: ymax]
                    mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                    mhd_image[mask_image != 1] = 0
                    mask_images.append(mask_image)
                    mhd_images.append(mhd_image)
                    # show_image(mhd_image)
                if not flag:
                    continue
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                # if width * height >= 900:
                #     patch_step = int(math.sqrt(width * height / 100))
                for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                    for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                        cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                    j - patch_size / 2: j + patch_size / 2, :]
                        if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                                   j - patch_size / 2: j + patch_size / 2, :]) / (
                                        (patch_size - 1) * (patch_size - 1) * 3)) < 0.7:
                            continue
                        save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        # print save_path
                        np.save(save_path, np.array(cur_patch))
                        patch_count += 1
                if patch_count == 1:
                    continue
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                    # roi_image.save(save_path)
        print count

    @staticmethod
    def extract_patch_npy_multiscale(dir_name, suffix_name, save_dir, patch_sizes, patch_step=1):
        '''
        提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式,提取多种尺度
        :param patch_sizes: 提取ｐａｔｃｈ的大小, array格式,[size1, size2] size1 > size2
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :param patch_step: 提取ｐａｔｃｈ的步长
        :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path)
                    mhd_image = np.squeeze(mhd_image)
                    mask_image = read_mhd_image(mask_path)
                    mask_image = np.squeeze(mask_image)
                    [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                    mask_image = mask_image[xmin: xmax, ymin: ymax]
                    mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                    mhd_image[mask_image != 1] = 0
                    mask_images.append(mask_image)
                    mhd_images.append(mhd_image)
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                if width * height >= 900:
                    patch_step = int(math.sqrt(width * height / 100))
                larget_patch_size = patch_sizes[0]
                small_patch_size = patch_sizes[1]
                for i in range(larget_patch_size / 2, width - larget_patch_size / 2, patch_step):
                    for j in range(larget_patch_size / 2, height - larget_patch_size / 2, patch_step):
                        larget_patch = mhd_images[i - larget_patch_size / 2:i + larget_patch_size / 2,
                                    j - larget_patch_size / 2: j + larget_patch_size / 2, :]
                        if (np.sum(mask_images[i - larget_patch_size / 2:i + larget_patch_size / 2,
                                   j - larget_patch_size / 2: j + larget_patch_size / 2, :]) / (
                                        (larget_patch_size - 1) * (larget_patch_size - 1) * 3)) < 0.9:
                            continue

                        small_patch = mhd_images[i - small_patch_size / 2: i + small_patch_size / 2,
                                      j - small_patch_size / 2: j + small_patch_size / 2, :]
                        cur_patch = [larget_patch, small_patch]
                        save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        np.save(save_path, np.array(cur_patch))
                        patch_count += 1
                if patch_count == 1:
                    print 'agnore'
                    continue
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                    # roi_image.save(save_path)
        print count

    @staticmethod
    def extract_liver_density(dir_name, suffix_name, save_dir, subclass, type):
        '''
        提取肝脏的平均密度
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :return: None
        '''
        count = 0

        names = os.listdir(dir_name)
        liver_density_dict = {}
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                liver_density = []
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    liver_mask_path = glob(os.path.join(dir_name, name, phasename + '*Liver*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path, rejust=False)
                    mhd_image = np.squeeze(mhd_image)
                    # show_image(mhd_image)
                    mask_image = read_mhd_image(mask_path)
                    mask_image = np.squeeze(mask_image)

                    liver_mask_image = read_mhd_image(liver_mask_path)
                    liver_mask_image = np.squeeze(liver_mask_image)
                    # 计算肝脏的平均密度
                    value_sum = np.sum(mhd_image[liver_mask_image != 0])
                    pixel_num = np.sum(liver_mask_image != 0)
                    average_value = (1.0 * value_sum) / (1.0 * pixel_num)
                    liver_density.append(average_value)
                print name, liver_density
                liver_density_dict[name] = liver_density
        scio.savemat(os.path.join(save_dir, 'raw_liver_density_' + str(subclass) + '_' + str(type) + '.mat'), liver_density_dict)
        return liver_density_dict

if __name__ == '__main__':

    # 提取ｐａｔｃｈ for BoVW
    for subclass in ['train', 'val', 'test']:
        for typeid in ['0', '1', '2', '3', '4']:
            # dir_name, suffix_name, save_dir, patch_size, patch_step=1
            ExtractPatch.extract_interior_patch_npy(
                dir_name='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/'+subclass,
                suffix_name=typeid,
                save_dir='/home/give/Documents/dataset/ICPR2018/BoVW-MI/patches/' + subclass + '/' + typeid,
                patch_size=8
            )
    '''
    # 提取ｐａｔｃｈ for deep learning
    for subclass in ['train', 'val', 'test']:
    # for subclass in ['0', '1', '2']:
        for typeid in ['0', '1', '2', '3', '4']:
            ExtractPatch.extract_patch_npy(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/'+subclass,
                typeid,
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/160_only_patch/' + subclass + '/' + typeid,
                patch_size=8
            )
    '''


    # 提取肝脏的平均密度
    # for subclass in ['train', 'val', 'test']:
    #     for typeid in ['0', '1', '2', '3', '4']:
    #          ExtractPatch.extract_liver_density(
    #              '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/' + subclass,
    #              typeid,
    #              save_dir='/home/give/PycharmProjects/MedicalImage/liver-density',
    #              subclass=subclass,
    #              type=typeid
    #          )

    # 提起多个尺度的ｐａｔｃｈ
    # for subclass in ['train', 'val', 'test']:
    # # for subclass in ['0', '1', '2']:
    #     for typeid in ['0', '1', '2', '3', '4']:
    #         ExtractPatch.extract_patch_npy_multiscale(
    #             '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/'+subclass,
    #             typeid,
    #             '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/ICIP/Frid-Adar/' + subclass + '/' + typeid,
    #             patch_sizes=[9, 5],
    #             patch_step=1
    #         )
