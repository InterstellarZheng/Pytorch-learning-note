import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_grad_enabled(True)  # Already on by default

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set
                                           , batch_size=100
                                           , shuffle=True  # to have the data reshuffled at every epoch
                                           )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # out_channels相当于kernel的个数
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)  # nn.Linear fully connected
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        # t = F.softmax(t, dim=1)
        return t


def get_num_correct(preds, labels):  # 返回预测正确数量
    return preds.argmax(dim=1).eq(labels).sum().item()


'''创建network'''
network = Network()
batch = next(iter(train_loader))
images, labels = batch

'''Calculating the Loss'''
# preds = network(images)
# # calculating the loss, 使用pytorch功能交叉熵损失函数，注意：损失函数中自带softmax因此在网络输出处不需要softmax
# loss = F.cross_entropy(preds, labels)
# # print(type(loss))
# # print(loss.item()) # 输出loss function 的值
# loss.backward()
# print(network.conv1.weight.grad.shape)  # 可以发现梯度张量和权重张量有着相同的形状
#
'''Updating the weights'''
# optimizer = optim.Adam(network.parameters(), lr=0.01)  # lr means Learning rate
# optimizer.step()  # updating the weights
# # optimizer = optim.SGD(network.parameters(),lr=0.01)

'''For one batch'''
# optimizer = optim.Adam(network.parameters(), lr=0.01)
# preds = network(images)  # pass batch
# loss = F.cross_entropy(preds, labels)  # calculate loss
# loss.backward()  # calculate gradients
# optimizer.step()  # update weights

'''For one epoch'''
# optimizer = optim.Adam(network.parameters(), lr=0.01)
#
# total_loss = 0
# total_correct = 0
#
# for batch in train_loader: # Get Batch
#     images, labels = batch
#
#     preds = network(images) # Pass Batch
#     loss = F.cross_entropy(preds, labels) # Calculate Loss
#
#     optimizer.zero_grad()   #每个batch需要将梯度置为0
#     loss.backward() # Calculate Gradients
#     optimizer.step() # Update Weights
#
#     total_loss += loss.item()
#     total_correct += get_num_correct(preds, labels)
#
# print(
#     "epoch:", 0,
#     "total_correct:", total_correct,
#     "loss:", total_loss
# )

'''The complete training loop'''
optimizer = optim.Adam(network.parameters(), lr=0.01)
for epoch in range(5):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:  # Get Batch
        images, labels = batch

        preds = network(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss

        optimizer.zero_grad()  # 每个batch需要将梯度置为0
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch:", epoch,
        "total_correct:", total_correct,
        "loss:", total_loss
    )

print(total_correct/len(train_set))