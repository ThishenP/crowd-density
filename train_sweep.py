from model import Net
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
from ray import tune
import pandas as pd

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
    now = datetime.now()
    run_start_datetime = now.strftime("%d-%m-%Y_%H-%M-%S")
    file = open(f"sweeps/sweep-{run_start_datetime}.txt", "w")
    file.close()

    file = open(f"sweeps/sweep-{run_start_datetime}.txt", "a")
    
    hyper(file)
    hyper(file, aspp = True)

        
def hyper(file, aspp=False):
    if aspp:
        file.write('aspp\n')
    else:
        file.write('basic\n')

    prev_configs = []
    num_trains =5
    for i in range(num_trains):
        config={
            "learning_rate": np.random.choice([1e-4, 2e-5 ,1e-5, 1e-7]),
            "batch_size": np.random.choice([1, 6, 8, 16]),
            "optimizer": np.random.choice(['adam', 'sgd'])
            }
        conf_str = str(config['learning_rate'])+" "+str(config['batch_size'])+" "+str(config['optimizer'])
        
        while conf_str in prev_configs:
            config={
            "learning_rate": np.random.choice([1e-4, 2e-5 ,1e-5, 1e-7]),
            "batch_size": np.random.choice([1, 6, 8, 16]),
            "optimizer": np.random.choice(['adam', 'sgd'])
            }
            conf_str = str(config['learning_rate'])+" "+str(config['batch_size'])+" "+str(config['optimizer'])
        print(i, config)
        val_mae_vals, train_losses = train(config, aspp)
        file.write(f"{i},{conf_str}")
        file.write("\n")
        for mae in val_mae_vals:
            file.write(f"{mae},") 
        file.write("\n")
        for los in train_losses:
            file.write(f"{los},") 
        file.write("\n")


    



def train(config, aspp):
    args = parse_args()
    wb = False ###############
    hp_sweep = True
    shut_down = False
    aspp = aspp
    save = False
    mae_vals = []
    train_losses = []

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


    #train_dataloader = DataLoader(train, batch_size = config["batch_size"], shuffle=True)#batch size of 1 since all different sizes
    train_dataloader = DataLoader(train, batch_size = int(config["batch_size"]), shuffle=True)#batch size of 1 since all different sizes
    val_dataloader = DataLoader(val, batch_size=1, shuffle=False)


    criterion = nn.MSELoss(reduction='sum').to(device)# same as nn.MSELoss(size_average=False)

    if config['optimizer'] == "adam":
        optimizer =optim.Adam(net.parameters(), lr=config["learning_rate"]) #lr = 1e-7
    else:
        optimizer = optim.SGD(net.parameters(), lr=config["learning_rate"], momentum=0.9)

    num_epochs = 400

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

        epoch_loss = running_loss/((int(len(train)/config["batch_size"]))+1)
        print(epoch_loss)
        train_losses.append(epoch_loss)
        if (epoch_loss >100000 or np.isnan(epoch_loss)):
            break

        if epoch%50 == 0: 
            losses =[]
        
            with torch.no_grad():    
                for i, data in enumerate(val_dataloader, 0):
                    # evaluate the model on the test set
                    inputs_val, labels_val = data
                    
                    
                    inputs_val = inputs_val.to(device)
                    labels_val = labels_val.to(device)
                    outputs_val = net(inputs)
                    
                    pred_count = np.sum(outputs_val[0][0].cpu().detach().numpy())
                    act_count = np.sum(labels_val[0].cpu().detach().numpy())
                    
                    losses.append(pred_count - act_count)

                    mae = np.mean(np.abs(losses))
                    mae_vals.append(mae)
    
        
    return mae_vals, train_losses
                

                
   
    


if __name__ == '__main__':
    main()