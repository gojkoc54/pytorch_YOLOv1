import torch
import torch.nn as nn

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        # To ensure that [1, 1024, 1, 1] will result in [1, 1204] so that the batch dimension doesn't get lost 
        return x.squeeze().view(-1, x.squeeze().shape[-1])
        # return x.squeeze()



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

