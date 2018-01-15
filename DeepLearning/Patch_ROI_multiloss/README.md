- 本目录下面的代码是使用Patch＆＆ROI一起训练得到的结果
- choose the alpha(when we extract patches, we use it)(patch size set to 7)
    - 80-models 是alpha（提取patch时的参数）为80时候的保存的参数
    - 120-models 是alpha（提取patch时的参数）为120时候的保存的参数
    - 140-models 是alpha（提取patch时的参数）为140时候的保存的参数
    - 160-models 是alpha（提取patch时的参数）为160时候的保存的参数
- choose the patch size(alpha set to 100)
    - models_3, the directory save the best parameters when the patch size set to 3.
    - models_5, the directory save the best parameters when the patch size set to 5.
    - models_7, the directory save the best parameters when the patch size set to 7.
    - models_9, the directory save the best parameters when the patch size set to 9.
    - models_11, the directory save the best parameters when the patch size set to 11.
    - models_13, the directory save the best parameters when the patch size set to 13.
- process:
    - execute train_imagenet_DIY.py, save the best parameters
    - execute val_DIY.py, see the accuracy of test patch dataset
    - execute generate_heatingmap.py, generate label map of training, validation and test dataset, respectively.
    - execute classification_heatingmap.py, get the instance-wise accuracy.