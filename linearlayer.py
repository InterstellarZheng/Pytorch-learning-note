import torch
import torch.nn as nn

'''利用简单的矩阵乘法来实现全连接层（线形层）'''
in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
], dtype=torch.float32)
# print(weight_matrix.matmul(in_features))


'''利用一层linear神经网络搭建'''
fc = nn.Linear(in_features=4, out_features=3, bias=True)
print(fc(in_features))
# 结果为随机数，由于神经网络初始参数为随机值

'''利用weight_matrix人为设定参数进行计算'''
fc.weight = nn.Parameter(weight_matrix)
print(fc(in_features))
# 与理想结果有差距，是因为神经网络建立的时候有bias项的干扰,改为bias = False即可

'''
可以看出fc作为全连接层的名称可以直接作为函数，是因为在module类中定义了特殊的函数__call__(),
同时__call__()函数又与forward()函数有关联
'''
