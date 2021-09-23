from darknet import Darknet
from torch.utils.data import random_split, TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch 

import matplotlib.pyplot as plt

def load_dataset(filepath, batch_size):
    
    input_size = (448, 448)
    
    transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.RandomRotation([90, 90]), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                    ])


    # Load the training set from the root folder:
    dataset = ImageFolder(root = filepath, transform = transform)

    # Create DataLoader objects for both the training and validation sets:  
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False) 

    return loader



def evaluate(model, data_loader, device):

    corrects = 0
    model.eval()

    for i, data in enumerate(data_loader, 0):

        input, target = data[0].to(device), data[1].to(device)
        
        pred = model(input)
        pred = pred if len(pred.shape) > 1 else pred.view((-1,) + tuple(pred.shape))

        curr_corrects = int(torch.sum(torch.argmax(pred, axis=1) == target))
        corrects += curr_corrects

        del data, input

    return float(corrects/len(data_loader.dataset)*100)


    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Darknet(fast_version=False, num_classes=2).to(device)

# checkpoint = torch.load('../checkpoints/Darknet_2_masks/interrupt_checkpoint_bs16.pt')
checkpoint = torch.load('../checkpoints/interrupt_checkpoint.pt')

model.load_state_dict(checkpoint['model_state_dict'])

batch_sizes = [1, 2, 4, 8, 16]

for bs in batch_sizes:
    loader = load_dataset(filepath = '../data/custom', batch_size = bs)
    acc = evaluate(model, loader, device)

    print("Batch size = {} ==>> ACC = {}%".format(bs, acc))

