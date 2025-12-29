import argparse


VOLUME_ROWS = 128
VOLUME_COLS = 160
VOLUME_DEPS = 128

NUM_CLASSES = 2

PATCH_SIZE_W = 128 # 裁剪的尺寸和输入网络的图像尺寸
PATCH_SIZE_H = 160

# VOLUME_ROWS = 128
# VOLUME_COLS = 128
# VOLUME_DEPS = 92

# NUM_CLASSES = 2

# PATCH_SIZE_W = 128 # 裁剪的尺寸和输入网络的图像尺寸
# PATCH_SIZE_H = 128

# VOLUME_ROWS = 240
# VOLUME_COLS = 240
# VOLUME_DEPS = 155

# NUM_CLASSES = 2

# PATCH_SIZE_W = 240 # 裁剪的尺寸和输入网络的图像尺寸
# PATCH_SIZE_H = 240

BATCH_SIZE = 16 # 一次输入多少图像进入网络
NUM_EPOCHS = 200



TRAIN_EXTRACTION_STEP = 12                 # 创建训练集提取的步长
TEST_EXTRACTION_STEP = 1      # 创建测试集提取的步长

# 路径设置
COM_CHOOSE = 3
# if COM_CHOOSE ==1:  # lsq
#     train_imgs_path = '/mnt/disk1/128x160x128/Train_Set'
#     un_train_imgs_path = '/mnt/disk1/128x160x128/Un_Train_Set'
#     val_imgs_path = '/mnt/disk1/128x160x128/Val_Set'
#     test_imgs_path =  '/mnt/disk1/128x160x128/Test_Set'
# if COM_CHOOSE == 2: # gwl
#     train_imgs_path = '/mnt/disk1/128x160x128/Train_Set'
#     un_train_imgs_path = '/mnt/disk1/128x160x128/Un_Train_Set'
#     val_imgs_path = '/mnt/disk1/128x160x128/Val_Set'
#     test_imgs_path =  '/mnt/disk1/128x160x128/Test_Set'
# if COM_CHOOSE == 3: # A222
#     train_imgs_path = '/mnt/disk1/128x160x128/Train_Set'
#     un_train_imgs_path = '/mnt/disk1/128x160x128/Un_Train_Set'
#     val_imgs_path = '/mnt/disk1/128x160x128/Val_Set'
#     test_imgs_path =  '/mnt/disk1/128x160x128/Test_Set'

# if COM_CHOOSE ==1:  # lsq
#     train_imgs_path = '/mnt/disk1/128x160x128_10/Train_Set'
#     un_train_imgs_path = '/mnt/disk1/128x160x128_10/Un_Train_Set'
#     val_imgs_path = '/mnt/disk1/128x160x128_10/Val_Set'
#     test_imgs_path =  '/mnt/disk1/128x160x128_10/Test_Set'
# if COM_CHOOSE == 2: # gwl
#     train_imgs_path = '/mnt/disk1/128x160x128_10/Train_Set'
#     un_train_imgs_path = '/mnt/disk1/128x160x128_10/Un_Train_Set'
#     val_imgs_path = '/mnt/disk1/128x160x128_10/Val_Set'
#     test_imgs_path =  '/mnt/disk1/128x160x128_10/Test_Set'
# if COM_CHOOSE == 3: # A222
#     train_imgs_path = '/mnt/disk1/128x160x128_10/Train_Set'
#     un_train_imgs_path = '/mnt/disk1/128x160x128_10/Un_Train_Set'
#     val_imgs_path = '/mnt/disk1/128x160x128_10/Val_Set'
#     test_imgs_path =  '/mnt/disk1/128x160x128_10/Test_Set'

# if COM_CHOOSE ==1:  # lsq
#     test_imgs_path = '/mnt/disk1/check_sup_data3/Test_Set'
# if COM_CHOOSE == 2: # gwl
#     test_imgs_path = '/mnt/disk1/check_sup_data3/Test_Set'
# if COM_CHOOSE == 3: # A222
#     test_imgs_path = '/mnt/disk1/check_sup_data3/Test_Set'


if COM_CHOOSE ==1:  # lsq
    train_imgs_path = '/mnt/disk1/hcp_data/Train_Set'
    un_train_imgs_path = '/mnt/disk1/hcp_data/Un_Train_Set'
    test_imgs_path =  '/mnt/disk1/hcp_data/Test_Set'
if COM_CHOOSE == 2: # gwl
    train_imgs_path = '/mnt/disk1/hcp_data/Train_Set'
    un_train_imgs_path = '/mnt/disk1/hcp_data/Un_Train_Set'
    test_imgs_path =  '/mnt/disk1/hcp_data/Test_Set'
if COM_CHOOSE == 3: # A222
    train_imgs_path = '/mnt/disk1/hcp_data/Train_Set'
    un_train_imgs_path = '/mnt/disk1/hcp_data/Un_Train_Set'
    test_imgs_path =  '/mnt/disk1/hcp_data/Test_Set'


MODEL_type = 2
if MODEL_type == 1:
    MODEL_name = 'model_ETC'      # 2D U-Net  for EC
elif MODEL_type == 2:
    MODEL_name = 'model_CTL'      # 2D Ournet for CL



# 是否选用多块GPU
FLAG_GPU = 1


