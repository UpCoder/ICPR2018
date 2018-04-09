class Config:
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    ROI_SIZE_W = 64
    ROI_SIZE_H = 64
    EXPAND_SIZE_W = 128
    EXPAND_SIZE_H = 128
    IMAGE_CHANNEL = 3
    phase_name = 'ART'
    MOMENTUM = 0.9
    ITERATOE_NUMBER = int(1e+4)
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 100
    DISTRIBUTION = [
        20,
        20,
        20,
        20,
        20
    ]
    color_maping = {
        0: [0, 255, 0],
        1: [0, 0, 255],
        2: [255, 0, 0],
        3: [0, 255, 255],
        4: [255, 255, 255]
    }
    divied_liver=True