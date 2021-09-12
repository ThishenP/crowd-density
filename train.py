from numpy.core.fromnumeric import resize
from model import Net
from dataset import CDEDataset
from torchvision import transforms

from PIL import Image
import torch
import os
import cv2

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def main():
    net = Net([(512,2),(512,2),(512,2),(256,4), (128,4), (64,4)], 23, 512)
    root_ims = 'ShanghaiTech/ShanghaiTech/part_A/train_data/images'
    root_ann = 'ShanghaiTech/ShanghaiTech/part_A/train_data/density_gt'
    im_list = os.listdir(root_ims)
    train = CDEDataset(im_list,root_ims,root_ann, transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

    root_ims = 'ShanghaiTech/ShanghaiTech/part_A/test_data/images'
    root_ann = 'ShanghaiTech/ShanghaiTech/part_A/test_data/density_gt'
    im_list = os.listdir(root_ims)
    test = CDEDataset(im_list,root_ims,root_ann, transform  = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

    

    train_dataloader = DataLoader(train, batch_size=1, shuffle=True) #batch size of 1 since all different sizes
    test_dataloader = DataLoader(test, batch_size=1, shuffle=True)

    criterion = nn.MSELoss(reduction='sum')# same as nn.MSELoss(size_average=False)
    optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)
    
    for epoch in range(3):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            print(outputs.shape)
            #quick fix idk if right:
            # ***********use this to preprocess - resizing is slow
            # print(labels.shape)
            # print(torch.sum(labels))
            # divisor = [labels.shape[2]/outputs.shape[2],labels.shape[3]/outputs.shape[3]]
            # resize_transform = transforms.Resize(size = (outputs.shape[2],outputs.shape[3]),interpolation = Image.BICUBIC)
            # labels = resize_transform(labels) #* divisor[0]*divisor[1]
            # print(torch.sum(labels))
            # print(labels.shape)

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            print(loss)
            print(loss.item())
            # print statistics
            running_loss += loss.item()
            if i % 4 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 4))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    main()