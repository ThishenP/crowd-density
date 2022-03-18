import torch.nn as nn
import torch
from torchvision import models, transforms
import torch.nn.functional as F

class BaseNet(nn.Module): #Base Net
    def __init__(self, arc, num_vgg_layers, num_channels_after_vgg):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
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

# class ASPPNet(nn.Module): #ASPP Net
#     def __init__(self, arc, num_vgg_layers, num_channels_after_vgg):
#         super().__init__()

#         vgg16 = models.vgg16(pretrained=True)
#         for param in vgg16.features.parameters():
#             param.require_grad = False
#         self.vgg_layers = vgg16.features[:num_vgg_layers]

        
#         self.aspp1 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 6 ,dilation = 6)
#         self.aspp2 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 12 ,dilation = 12)
#         self.aspp3 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 18 ,dilation = 18)
#         self.aspp4 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 24 ,dilation = 24)

#         self.dilated_layers = create_layers(arc, num_channels_after_vgg) #in size may change depending on number of vgg layers
#         self.last = nn.Conv2d(arc[-1][0], 1, kernel_size=1)
        
                
#     def forward(self, image):
#         vgg_out = self.vgg_layers(image)
        
#         x1 = self.aspp1(vgg_out)
#         x2 = self.aspp2(vgg_out)
#         x3 = self.aspp3(vgg_out)
#         x4 = self.aspp4(vgg_out)
#         out = torch.cat((x1, x2, x3, x4), dim=1)

#         print(out.shape)

#         out = self.dilated_layers(out)
#         out = self.last(out)

#         return out

# ASPP code leveraged from https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class ASPPNet(nn.Module): #ASPP Net
    def __init__(self, arc, num_vgg_layers, num_channels_after_vgg):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.require_grad = False
        self.vgg_layers = vgg16.features[:num_vgg_layers]

        
        self.aspp = ASPP(num_channels_after_vgg, [6, 12, 18], arc[0][0])

        self.dilated_layers = create_layers(arc, num_channels_after_vgg) #in size may change depending on number of vgg layers
        self.last = nn.Conv2d(arc[-1][0], 1, kernel_size=1)
        
                
    def forward(self, image):
        vgg_out = self.vgg_layers(image)
        out = self.aspp(vgg_out)
        out = self.dilated_layers(out)
        out = self.last(out)

        return out

def create_layers(arc, in_size):
    layers = []
    layers.append(nn.Conv2d(in_size, arc[0][0], kernel_size=3, padding= arc[0][1] ,dilation = arc[0][1]))
    layers.append(nn.ReLU(inplace = True))
    for i in range(1, len(arc)):
        layers.append(nn.Conv2d(arc[i-1][0], arc[i][0], kernel_size=3, padding= 2 ,dilation = arc[i][1]))
        layers.append(nn.ReLU(inplace = True))
    return nn.Sequential(*layers)
