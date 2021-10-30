import torch.nn as nn
import torch
from torchvision import models, transforms

class Net(nn.Module):
    def __init__(self, arc, num_vgg_layers, num_channels_after_vgg, aspp = False):
        super().__init__()
        self.aspp = aspp

        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.require_grad = False
        self.vgg_layers = vgg16.features[:num_vgg_layers]

        if self.aspp:
            self.aspp1 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 2 ,dilation = 2)
            self.aspp2 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 4 ,dilation = 4)
            self.aspp3 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 8 ,dilation = 8)
            self.aspp4 = nn.Conv2d(num_channels_after_vgg, int(arc[0][0]/4), kernel_size=3, padding= 16 ,dilation = 16)

        self.dilated_layers = create_layers(arc, num_channels_after_vgg) #in size may change depending on number of vgg layers
        self.last = nn.Conv2d(arc[-1][0], 1, kernel_size=1)
        
                
    def forward(self, image):
        vgg_out = self.vgg_layers(image)
        if self.aspp:
            x1 = self.aspp1(vgg_out)
            x2 = self.aspp2(vgg_out)
            x3 = self.aspp3(vgg_out)
            x4 = self.aspp4(vgg_out)
            out = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            out = vgg_out
        
        
        out = self.dilated_layers(out)
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