import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer

import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(14*14*64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=2)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)


    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 98.64
    #optimizer = optim.Adam(net.parameters(), lr=0.001) # acc = 99.30
    optimizer = torch_optimizer.RAdam(net.parameters(), lr=0.001) # acc = 99.32

    net = net.train()

    for epoch in range(20):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print('%d, %d, loss: %.3f' % (epoch+1, i+1, running_loss/100))
                running_loss = 0.0

    print('Finish')

    correct = 0
    total = 0
    net = net.eval()
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy: %.2f' % (correct/total * 100))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Elapse time: %.f [sec]' % (time.time() - start_time))
