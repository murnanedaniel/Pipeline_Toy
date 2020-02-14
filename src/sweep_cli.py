import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from models.convNet import Net
import yaml

import matplotlib.pyplot as plt
import numpy as np

import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('sweep_cli.py')
    add_arg = parser.add_argument
    add_arg('--hidden_dim_1', type=int)
    add_arg('--hidden_dim_2', type=int)
    add_arg('--hidden_dim_3', type=int)
    add_arg('--hidden_dim_4', type=int)
    add_arg('--kern_1', type=int)
    add_arg('--kern_2', type=int)
    add_arg('--lr', type=float)
    add_arg('--momentum', type=float)
    add_arg('--n_epochs', type=int)
#     add_arg('input', default='./data', help="The directory of image training data")
#     add_arg('output', default='./cifar_net.pth', help="The path to save model data to")

    return parser.parse_args()

def load_data(input_dir):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root=input_dir, train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=input_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

def main():
        
    # Parse the command line
    args = parse_args()
    
#     for k in vars(args):
#         print(k, vars(args)[k])
    
    input_dir = "data"
#     args.input
    output_dir = "test-output.pth"
#     args.output

    
    # Set up training and evaluation datasets
    trainloader, testloader = load_data(input_dir)
    
    # Initialise the model and monitoring
    wandb.init(project="convnet-toy")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using ", device)
    
    m_dic = ["kern_1", "hidden_dim_1", "kern_2", "hidden_dim_2", "hidden_dim_3", "hidden_dim_4"] # Define the HPs given by W&B
    m_configs = {k: vars(args)[k] for k in m_dic} # Retrieve the HPs from W&B
    m_configs = {**m_configs, 'output_dim': 10} # Manually specify any HPs not part of the sweep

    model = Net(**m_configs).to(device) # Initialise the model, and send it to the GPU
    
    criterion = nn.CrossEntropyLoss()
    o_dic = ["lr", "momentum"]
    o_configs = {k: vars(args)[k] for k in o_dic} 
    optimizer = optim.SGD(model.parameters(), **o_configs)
    
    wandb.watch(model)
    
    for epoch in range(vars(args)["n_epochs"]):  # loop over the dataset multiple times

        model.train()
        running_loss = 0.0
        train_loss, val_loss = 0., 0.
        for i, data in enumerate(trainloader, 0):

            # zero the parameter gradients
            optimizer.zero_grad()

            # send the data to the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        model.eval()
        correct, total = 0, 0
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()

        val_acc = 100 * correct / total
        wandb.log({"train loss": train_loss, "val loss": val_loss, "val accuracy": val_acc})
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            val_acc))
        
    print('Finished Training')
    
 
    torch.save(model.state_dict(), output_dir)

if __name__ == '__main__':
    main()