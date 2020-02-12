import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
from models.convNet import Net

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('input', default='./data', help="The directory of image training data")
    add_arg('model', default='./cifar_net.pth', help="The path to load model data from")
    add_arg('metric_file', default='acc.metric', help="The path to save the accuracy of this file to")

    return parser.parse_args()
    
def main():
    
    # Parse the command line
    args = parse_args()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    input_dir = args.input
    
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
    
    model_dir = args.model
    model.load_state_dict(torch.load(model_dir))

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    tic = time.time()
    
    model.eval()
    correct, total = 0, 0
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    toc = time.time()
    
    acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % acc )
        
    metrics = np.asarray([acc, toc - tic ])
    metric_file = args.metric_file
    np.savetxt(metric_file, metrics)
    
if __name__ == '__main__':
    main()