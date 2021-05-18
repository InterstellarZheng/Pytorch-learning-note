import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set
                                           , batch_size=10
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
        # (1) input layer
        t = t

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


'''image prediction without train'''
torch.set_grad_enabled(False)  # 由于没有训练内容，不需要使用计算图，可以关闭减小内存
network = Network()
# sample = next(iter(train_set))
# image,label = sample
# pred = network(image.unsqueeze(0))
# print(pred)
# print(pred.shape)
# print(pred.argmax(dim=1))


'''batch of images prediction without train'''
batch =next(iter(train_loader))
images,labels = batch
preds = network(images) # DataLoader自带batch通道，不需要unsqueeze
print(preds)
print(preds.shape)
print(preds.argmax(dim=1))
