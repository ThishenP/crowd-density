from model import BaseNet, ASPPNet, SkipASPPNet
from dataset import CDEDataset
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
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
    if torch.cuda.is_available(): 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    #net = SkipASPPNet(16, 256).to(device)
    #train(net, 'skip-aspp', lr = 1e-6, batch_size=6, epochs = 2000,save = True, wb = True)
    for i in range(10):
        #no dilation
        net = BaseNet([(256,1),(256,1), (128,1),(128,1), (64,1), (64,1)], 16, 256).to(device)
        train(net, 'conv', lr = 5e-5, batch_size=8, epochs = 800,save = True, wb = True)

        #dilation = 2
        #net = BaseNet([(256,2),(256,2), (128,2),(128,2), (64,2), (64,2)], 16, 256).to(device)
        #train(net, 'base', lr = 2e-5, batch_size=8, epochs = 800,save = True, wb = True)    

#        net = ASPPNet([(256,2),(256,2), (128,2),(128,2), (64,2), (64,2)], 16, 256).to(device)
#        train(net, 'aspp', lr = 2e-5, batch_size=8, epochs = 800,save = True, wb = True)


def train(net, model_name, lr=1e-6, batch_size=6, epochs = 800, wb = False, shut_down = False, save = False):
    print(f"training {model_name}")
    if torch.cuda.is_available(): 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if wb:
        wandb.init(project="cde", entity="thishen")
        config = wandb.config

    root_ims = '../CDE_Data/train/images'
    root_ann = '../CDE_Data/train/density_gt'

    im_list = os.listdir(root_ims)
    
    train = CDEDataset(im_list,root_ims,root_ann, transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]), dilated2 = False)

    root_ims = '../CDE_Data/val/images'
    root_ann = '../CDE_Data/val/density_gt'

    im_list = os.listdir(root_ims)

    val = CDEDataset(im_list,root_ims,root_ann, transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]), train = False)



    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)#batch size of 1 since all different sizes
    val_dataloader = DataLoader(val, batch_size=1, shuffle=False)


    criterion = nn.MSELoss(reduction='sum').to(device)# same as nn.MSELoss(size_average=False)
    
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) #lr = 1e-7
    optimizer = optim.Adam(net.parameters(), lr=lr) # lr=1e-6 for skip

    num_epochs = epochs

    if save:
        now = datetime.now()
        run_start_datetime = now.strftime("%d-%m-%Y_%H-%M-%S")
        os.mkdir(f'models/{run_start_datetime}-{model_name}')
        os.mkdir(f'../checkpoints/{run_start_datetime}-{model_name}')

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

            running_loss += loss.item()

        epoch_loss = running_loss/((int(len(train)/batch_size))+1)
        print(epoch_loss)

        if wb:
            wandb.log({"loss": epoch_loss})


        if epoch%200 == 0: 
            val_losses = []
            with torch.no_grad():    
                for i, data in enumerate(val_dataloader, 0):
                    # evaluate the model on the test set
                    inputs_val, labels_val = data

                    inputs_val = inputs_val.to(device)
                    labels_val = labels_val.to(device)
                    outputs_val = net(inputs_val)
                    
                    pred_count = np.sum(outputs_val[0][0].cpu().detach().numpy())
                    act_count = np.sum(labels_val[0].cpu().detach().numpy())
                    
                    val_losses.append(pred_count - act_count)

                mae = np.mean(np.abs(val_losses))
                if wb:
                    wandb.log({"mae-on-validation": mae})
            if save:
                torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

                }, f"../checkpoints/{run_start_datetime}-{model_name}/checkpoint_at_{epoch}.pt")
                torch.save(net, f"models/{run_start_datetime}-{model_name}/model_at_{epoch}.pt")
    if save:
        torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

                }, f"../checkpoints/{run_start_datetime}-{model_name}/checkpoint_at_{epoch}.pt")
        torch.save(net, f"models/{run_start_datetime}-{model_name}/model_at_{num_epochs}.pt")    
    print('Finished Training')
    
    if shut_down:
        os.system('shutdown -s')

if __name__ == '__main__':
    main()
