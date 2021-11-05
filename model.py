import torch.nn as nn
import torch
from torchvision import models, transforms

class BaseNet(nn.Module):
    def __init__(self, arc, num_vgg_layers, num_channels_after_vgg):
        super().__init__()

        vgg16 = models.vgg16_bn(pretrained=True)
        for param in vgg16.features.parameters():
            param.require_grad = False
        self.vgg_layers = vgg16.features[:num_vgg_layers]
        self.dilated_layers = create_layers(arc, num_channels_after_vgg) #in size may change depending on number of vgg layers
        self.last = nn.Conv2d(arc[-1][0], 1, kernel_size=1)
        
                
    def forward(self, image):
        vgg_out = self.vgg_layers(image)
        out = self.dilated_layers(vgg_out)
        out = self.last(out)

        return out

class ASPPNet(nn.Module):
    def __init__(self, arc, num_vgg_layers, num_channels_after_vgg):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.require_grad = False
        self.vgg_layers = vgg16.features[:num_vgg_layers]

        
        self.aspp1 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 6 ,dilation = 6)
        self.aspp2 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 12 ,dilation = 12)
        self.aspp3 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 18 ,dilation = 18)
        self.aspp4 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 24 ,dilation = 24)

        self.dilated_layers = create_layers(arc, num_channels_after_vgg) #in size may change depending on number of vgg layers
        self.last = nn.Conv2d(arc[-1][0], 1, kernel_size=1)
        
                
    def forward(self, image):
        vgg_out = self.vgg_layers(image)
        
        x1 = self.aspp1(vgg_out)
        x2 = self.aspp2(vgg_out)
        x3 = self.aspp3(vgg_out)
        x4 = self.aspp4(vgg_out)
        out = torch.cat((x1, x2, x3, x4), dim=1)

        out = self.dilated_layers(out)
        out = self.last(out)

        return out

class SkipASPPNet(nn.Module):
    def __init__(self, num_vgg_layers, num_channels_after_vgg):
        super().__init__()


        vgg16 = models.vgg16(pretrained=True)

        for param in vgg16.features.parameters():
            param.require_grad = False
        self.vgg_layers = vgg16.features[:num_vgg_layers]

        
        self.aspp1 = nn.Conv2d(num_channels_after_vgg, 64, kernel_size=3, padding= 6 ,dilation = 6)
        self.aspp2 = nn.Conv2d(num_channels_after_vgg, 64, kernel_size=3, padding= 12 ,dilation = 12)
        self.aspp3 = nn.Conv2d(num_channels_after_vgg, 64, kernel_size=3, padding= 18 ,dilation = 18)
        self.aspp4 = nn.Conv2d(num_channels_after_vgg, 64, kernel_size=3, padding= 24 ,dilation = 24)

        self.dilated1 = nn.Conv2d(256,256,kernel_size = 3, padding =2, dilation = 2)
        self.dilated2 = nn.Conv2d(256,128,kernel_size = 3, padding =2, dilation = 2)
        self.dilated3 = nn.Conv2d(128+256,128,kernel_size = 3, padding =2, dilation = 2)
        self.dilated4 = nn.Conv2d(128,128,kernel_size = 3, padding = 2, dilation = 2)
        self.dilated5 = nn.Conv2d(128+128,64,kernel_size = 3, padding = 2, dilation = 2)
        self.dilated6 = nn.Conv2d(64,64,kernel_size = 3, padding = 2, dilation = 2)

       # self.dilated_layers = create_layers(arc, num_channels_after_vgg) #in size may change depending on number of vgg layers
        self.last = nn.Conv2d(64, 1, kernel_size=1)
        
                
    def forward(self, image):
        #print(image.shape)
        vgg_out = self.vgg_layers(image)
        #print('vgg:',vgg_out.shape)
        x1 = self.aspp1(vgg_out)
        x2 = self.aspp2(vgg_out)
        x3 = self.aspp3(vgg_out)
        x4 = self.aspp4(vgg_out)
        cat = torch.cat((x1, x2, x3, x4), dim=1)

        out = self.dilated1(cat)
        skip = self.dilated2(out)
        out = self.dilated3(torch.cat((skip,cat), dim=1))
        out = self.dilated4(out)
        out = self.dilated5(torch.cat((out,skip), dim=1))
        out = self.dilated6(out)
        
        out = self.last(out)

        return out

def create_layers(arc, in_size):
    layers = []
    layers.append(nn.Conv2d(in_size, arc[0][0], kernel_size=3, padding= arc[0][1] ,dilation = arc[0][1]))
    layers.append(nn.ReLU(inplace = True))
    for i in range(1, len(arc)):
        layers.append(nn.Conv2d(arc[i-1][0], arc[i][0], kernel_size=3, padding= 2 ,dilation = arc[i][1]))
        layers.append(nn.ReLU(inplace = True)) #check reason for inplace
    return nn.Sequential(*layers)