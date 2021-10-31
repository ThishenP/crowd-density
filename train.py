from model import Net
from dataset import CDEDataset
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datetime import datetime
import wandb

#FIX ARGS not currenlty in use
def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--wb', default=False,
                        help='Use Weights and Biases')
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    wb = False ###############
    shut_down = False
    aspp = False
    save = False
    hyper_params = [""]

    if wb:
        wandb.init(project="cde", entity="thishen")
        config = wandb.config
        config.learning_rate = 1e-7

    if torch.cuda.is_available(): 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    device


    net = Net([(256,2),(256,2), (128,2),(128,2), (64,2), (64,2)], 16, 256, aspp=aspp).to(device)

    root_ims = '../CDE_Data/train/images'
    root_ann = '../CDE_Data/train/density_gt'
    im_list = os.listdir(root_ims)

    
    train = CDEDataset(im_list,root_ims,root_ann, transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))

    root_ims = '../CDE_Data/val/images'
    root_ann = '../CDE_Data/val/density_gt'
    im_list = os.listdir(root_ims)

    val = CDEDataset(im_list,root_ims,root_ann, transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]), train = False)



    train_dataloader = DataLoader(train, batch_size=6, shuffle=True)#batch size of 1 since all different sizes
    val_dataloader = DataLoader(val, batch_size=1, shuffle=False)


    criterion = nn.MSELoss(reduction='sum').to(device)# same as nn.MSELoss(size_average=False)
    
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9) #lr = 1e-7

    num_epochs = 800

    if save:
        now = datetime.now()
        
        run_start_datetime = now.strftime("%d-%m-%Y_%H-%M-%S")
        os.mkdir(f'models/{run_start_datetime}')
        os.mkdir(f'../checkpoints/{run_start_datetime}')

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


        print(running_loss/48)
        if wb:
            wandb.log({"loss": running_loss/48})
        if epoch%50 == 0: 
            if save:
                torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

                }, f"../checkpoints/{run_start_datetime}/checkpoint_at_{epoch}.pt")
                torch.save(net, f"models/{run_start_datetime}/model_at_{epoch}.pt")
    if save:
        torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

                }, f"../checkpoints/{run_start_datetime}/checkpoint_at_{epoch}.pt")
        torch.save(net, f"models/{run_start_datetime}/model_at_{num_epochs}.pt")    
    print('Finished Training')
    
    if shut_down:
        os.system('shutdown -s')

if __name__ == '__main__':
    main()
