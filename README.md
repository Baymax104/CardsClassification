# 基于CNN的扑克牌图像分类

## 数据集介绍

数据集源自Kaggle的公开数据集[Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

数据集放置在根目录下，目录形式如下

- ML/dataset
  - test
    - ace of clubs
    - ...
  - train
    - ace of clubs
    - ...
  - valid
    - ace of clubs
    - ...
  - cards.csv

其中标签有53类，训练集图像7624张，验证集图像265张，测试集图像265张，类别分布较均衡

图像大小均为224\*224，包含RGB三通道

对图像进行归一化和转置后得
- 训练集Tensor(7624, 3, 224, 224)
- 验证集Tensor(265, 3, 224, 224)
- 测试集Tensor(265, 3, 224, 224)


## 网络构建


## 训练结果



