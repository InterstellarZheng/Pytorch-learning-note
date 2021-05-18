import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

'''1.数据集建立，变换和导入ETL principal: E(extract)    T(transform)    L(load)'''
train_set = torchvision.datasets.FashionMNIST(
    # 数据存放位置
    root='./data/FashionMNIST'
    # 训练参数，True数据用于训练   FashionMNIST中6万张用作训练数据，1万张用于测试数据
    , train=True
    # 下载参数，如果指定目录下没有数据集则下载数据
    , download=True
    # torchvision.transforms.Compose()为类，将transforms列表里面的transform操作进行遍历    图片转换为tensor
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set
                                           , batch_size=1000
                                           , shuffle=True  # to have the data reshuffled at every epoch
                                           )

'''2.对train_set进行深入了解'''
print(len(train_set))  # 训练集个数
print(train_set.targets)  # 标签
print(train_set.targets.bincount())  # 检测数据集是否为平衡数据集，即各类别图象数相同

sample = next(iter(train_set))  # iter将可迭代对象转换为迭代器，并通过next函数进行读取，可以借此来查看数据集的数据形式
print(len(sample))  # 返回一个数据对应的数据长度，长度为2，分别为图像数据张量以及相应的标签张量
print(type(sample))  # <class 'tuple'>
image, label = sample  # 序列解压缩
'''
相当于
image = sample[0]
label = sample[1]
'''

print(image.shape)  # torch.Size([1, 28, 28])
print(type(label))  # <class 'int'>
plt.imshow(image.squeeze(), cmap='gray')
# plt.show()
print('label', label)

'''3.work with batches and dataloader'''
batch = next(iter(train_loader))
print(len(batch))
print(type(batch))  # <class 'list'>
images, labels = batch
print(images.shape)
print(labels.shape)
