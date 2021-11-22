import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)
              

class FlowRefNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(FlowRefNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.fea_conv0 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.featureExtractor = make_layer(ResidualBlock, nf=64, n_layers=4)
        self.fea_conv_last = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

    def forward(self, input1,input2,input3):
        input = torch.cat([input1, input2, input3],dim=1) 
        out = self.relu(self.fea_conv0(input)) 
        out = self.featureExtractor(out) 
        out = self.fea_conv_last(out) 
        return out


class ViewRefNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(ViewRefNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.fea_conv0 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.featureExtractor = make_layer(ResidualBlock, nf=64, n_layers=4)
        self.fea_conv_last = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

    def forward(self, input1,input2,input3):
        input = torch.cat([input1, input2, input3],dim=1) 
        out = self.relu(self.fea_conv0(input)) 
        out = self.featureExtractor(out) 
        out = self.fea_conv_last(out) 
        return out + input1 