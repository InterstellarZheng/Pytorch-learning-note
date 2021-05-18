import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)  # relu and pooling have no weights
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)
        return t

    # # 修改print的输出内容          overrides method
    # def __repr__(self):
    #     return "MyNetwork"

    ''' hyperparameter
    The parameter in_channels and out channels et. are 'hyperparameters' which are choosen by users
    while the in_channels in conv1 and out_fearures in out are 'data dependent hyperparameters' which
    determined by the dataset and lable number
    '''
    '''Learnable parameters
    With learnable parameters, we typically start out with a set of arbitrary values, and these values 
    then get updated in an iterative fashion as the network learns, which specifically mean that the 
    network is learning the appropriate values for the learnable parameters. Appropriate values are 
    values that minimize the loss function.
    '''


network = Network()
# print(network)
# print(network.conv1.weight)

# print(network.conv1.weight.shape)
# print(network.conv2.weight.shape)
# print(network.fc1.weight.shape)
# print(network.fc2.weight.shape)
# print(network.out.weight.shape)

'''快速查看参数tensor'''
for param in network.parameters():
    print(param.shape)

for name, param in network.named_parameters():
    print(name, '\t\t', param.shape)
