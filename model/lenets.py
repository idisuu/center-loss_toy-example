import torch.nn as nn

import numpy as np

class LeNetsPlus(nn.Module):

    def __init__(self):
        super(LeNetsPlus, self).__init__()

        self.maxpool = nn.MaxPool2d(2)

        self.conv_1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu_1_1 = nn.PReLU()
        self.conv_1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu_1_2 = nn.PReLU()

        self.conv_2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu_2_1 = nn.PReLU()
        self.conv_2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu_2_2 = nn.PReLU()

        self.conv_3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu_3_1 = nn.PReLU()
        self.conv_3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu_3_2 = nn.PReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_extractor = nn.Linear(128, 2)
        self.prelu_feature_extractor = nn.PReLU()
        self.linear = nn.Linear(2, 10)

        # initialize weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity="relu")
            elif isinstance(nn.BatchNorm2d, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Caculate model parameters
        self.total_params = 0
        for x in filter(lambda p: p.requires_grad, self.parameters()):
            self.total_params += np.prod(x.data.numpy().shape)

        print(f"Total parameters: {self.total_params}")

    def forward(self, x):
        out = self.conv_1_1(x)
        out = self.prelu_1_1(out)
        out = self.conv_1_2 (out)
        out = self.prelu_1_2(out)
        out = self.maxpool(out)

        out = self.conv_2_1(out)
        out = self.prelu_2_1(out)
        out = self.conv_2_2(out)
        out = self.prelu_2_2(out)
        out = self.maxpool(out)

        out = self.conv_3_1(out)
        out = self.prelu_3_1(out)
        out = self.conv_3_2(out)
        out = self.prelu_3_2(out)

        out = self.avgpool(out)
        
        feature = self.feature_extractor(out.view(-1, 128))
        feature = self.prelu_feature_extractor(feature)

        out = self.linear(feature)
        
        return feature, out