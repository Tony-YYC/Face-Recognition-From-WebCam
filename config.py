# -*- encoding: utf8 -*-
import os
PROJECT_PATH = os.getcwd()
#print(PROJECT_PATH)

# 训练数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data\\train")
#print(DATA_TRAIN)
# 验证数据集
DATA_TEST = os.path.join(PROJECT_PATH, "data\\test")
# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data\\model")
DEFAULT_MODEL = "checkpoint_model_epoch.pth.tar"
EPOCHS = 10
BATCH_SIZE = 5
FONTS = os.path.join(PROJECT_PATH, "Fonts\\simsun.ttc")
NAME_TXT = os.path.join(PROJECT_PATH,"name.txt")