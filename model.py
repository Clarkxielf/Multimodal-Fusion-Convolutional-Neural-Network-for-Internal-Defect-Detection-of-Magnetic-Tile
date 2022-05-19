import torch
import torch.nn as nn
import time


class AlexNet_3D(nn.Module):
    def __init__(self,init_weights=True):
        super(AlexNet_3D,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))   #outsize=[batch,256,3,3]




        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)


        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AlexNet_1D(nn.Module):
    def __init__(self,init_weights=True):
        super(AlexNet_1D,self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(9)    #outsize=[batch,256,9]



        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.features(x.unsqueeze(1))
        x = self.avgpool(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class SqueezeExcitation1(nn.Module):
    def __init__(self):
        super(SqueezeExcitation1, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Conv2d(256,64,1)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Conv2d(64,256,1)
        self.sigmoid=nn.Sigmoid()


    def forward(self,x):
        scale=self.avgpool(x)
        scale=self.fc1(scale)
        scale=self.relu(scale)
        scale=self.fc2(scale)
        scale=self.sigmoid(scale)
        x=scale*x

        return x

class SqueezeExcitation2(nn.Module):
    def __init__(self):
        super(SqueezeExcitation2, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.fc1=nn.Conv1d(256,64,1)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Conv1d(64,256,1)
        self.sigmod=nn.Sigmoid()


    def forward(self,x):
        scale=self.avgpool(x)
        scale=self.fc1(scale)
        scale=self.relu(scale)
        scale=self.fc2(scale)
        scale=self.sigmod(scale)
        x=scale*x

        return x

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention,self).__init__()
        self.maxpool=nn.AdaptiveMaxPool1d(1)
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        # self.sigmoid=nn.Sigmoid()
        self.conv1=nn.Conv1d(10,16,kernel_size=1,stride=1)

    def forward(self,x,y):
        max_out=self.maxpool(y)
        x=torch.cat([x,max_out],dim=2)
        x=x.transpose(1,2)
        x=self.conv1(x)
        x=x.transpose(1,2)

        return x
class FuseNet(nn.Module):
    def __init__(self,num_class=2,init_weights=True):
        super(FuseNet,self).__init__()

        self.branch1 = AlexNet_1D()
        self.branch2 = AlexNet_1D()
        self.branch3 = AlexNet_3D()
        self.se=SqueezeExcitation2()
        self.ca=CrossAttention()
        self.classfier = nn.Sequential(
            nn.Linear(12288, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class)
        )


        if init_weights:
            self._initialize_weights()

    def forward(self,x,y,z):

        x = self.branch1(x.contiguous())
        y = self.branch2(y.contiguous())
        z = self.branch3(z.contiguous())
        z=z.view(z.size(0),z.size(1),-1)
        x=self.se(x)
        y=self.se(y)
        z=self.se(z)
        x=self.ca(x,y)
        y=self.ca(y,z)
        z=self.ca(z,x)
        xyz = torch.cat([x, y, z], dim=1)
        xyz=xyz.view(xyz.size(0),-1)
        out_put = self.classfier(xyz)

        return out_put


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# model=FuseNet()
#
# x=torch.randn(1,7000)
# y=torch.randn(1,7000)
# z=torch.randn(1,3,320,320)
# start_time = time.time()
# for i in range(10):
#     redict = model(x,y,z)
# end_time = time.time()
# sum_time = end_time - start_time
# print(sum_time/10)
# # print(model(x,y,z))

