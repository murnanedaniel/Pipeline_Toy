import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from models.convNet import Net

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('input', default='./data', help="The directory of image training data")
    add_arg('output', default='./cifar_net.pth', help="The path to save model data to")

    return parser.parse_args()

def main():
    
    # Parse the command line
    args = parse_args()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    input_dir = args.input
    
    trainset = torchvision.datasets.CIFAR10(root=input_dir, train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=input_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Using ", device)
    # model = Edge_Class_Net( input_dim=2, hidden_dim=64, n_graph_iters=4).to(device)
    model = Net().to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    
    for epoch in range(2):  # loop over the dataset multiple times

        model.train()
        running_loss = 0.0
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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        
    print('Finished Training')
    
    output_dir = args.output
    torch.save(model.state_dict(), output_dir)
    
if __name__ == '__main__':
    main()