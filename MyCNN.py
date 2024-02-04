import torch.nn as nn
import torch

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=16, stride=1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    

        self.fc1 = nn.Linear(in_features=148, out_features=64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(in_features=64, out_features=4)
        
    def forward(self, image, direction):
        x = self.convolution1(image)
        x = self.relu1(x)
        x = self.pool1(x)
        
        flattened_x = x.view(x.size(0), -1)
        # Concatenate the flattened image tensor with the direction tensor
        x = torch.cat([flattened_x, direction], dim=1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.output(x)
       
        return x
