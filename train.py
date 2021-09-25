from model import Net
from dataset import CDEDataset
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main():

    if torch.cuda.is_available(): 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    device


    net = Net([(512,2),(512,2),(512,2),(256,2), (128,2), (64,2)], 23, 512)

    root_ims = 'ShanghaiTech/ShanghaiTech/part_A/train_data/images'
    root_ann = 'ShanghaiTech/ShanghaiTech/part_A/train_data/density_gt'
    im_list = os.listdir(root_ims)
    train = CDEDataset(im_list,root_ims,root_ann, transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))

    root_ims = 'ShanghaiTech/ShanghaiTech/part_A/test_data/images'
    root_ann = 'ShanghaiTech/ShanghaiTech/part_A/test_data/density_gt'
    im_list = os.listdir(root_ims)
    test = CDEDataset(im_list,root_ims,root_ann, transform  = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))

    train_dataloader = DataLoader(train, batch_size=1, shuffle=True)#batch size of 1 since all different sizes
    test_dataloader = DataLoader(test, batch_size=1, shuffle=True)

    criterion = nn.MSELoss(reduction='sum').to(device)# same as nn.MSELoss(size_average=False)
    optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)

    num_epochs = 10

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(running_loss/300)

    print('Finished Training')


if __name__ == '__main__':
    main()
