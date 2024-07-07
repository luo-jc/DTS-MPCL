import torch.nn as nn


class MPCL(nn.Module):
    def __init__(self):
        super(MPCL, self).__init__()
        self.conv_class_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_class_2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv_class_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_class_4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(1)
        self.LRelu = nn.LeakyReLU(inplace=True)

        self.liner_1 = nn.Linear(1024, 1024)
        self.liner_2 = nn.Linear(1024, 512)
        self.liner_3 = nn.Linear(512, 256)
        self.liner_4 = nn.Linear(256, 128)
        self.liner_5 = nn.Linear(128, 5)

    def forward(self, x):
        x1 = self.LRelu(self.conv_class_1(x))
        x1 = self.max_pool(x1)
        x1 = self.LRelu(self.conv_class_2(x1))
        x1 = self.max_pool(x1)
        x2 = self.LRelu(self.conv_class_3(x1))
        x2 = self.max_pool(x2)
        x3 = self.batch_norm(self.conv_class_4(x2))

        x3 = x3.view(x.size(0), -1)

        x4 = self.LRelu(self.liner_1(x3))
        x4 = self.LRelu(self.liner_2(x4))
        x4 = self.LRelu(self.liner_3(x4))
        x5 = self.LRelu(self.liner_4(x4))
        x5 = self.liner_5(x5)
        return x5
