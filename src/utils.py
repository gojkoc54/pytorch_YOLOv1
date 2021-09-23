import argparse
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import sys
import numpy as np


# Since argparse has problems with bool values (parses them as strings)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




# ========================================================================================
# === DATA LOADING ===
# ========================================================================================

def load_datasets(train_filepath, val_filepath, input_size, batch_size, cifar_10=False, num_workers=0):
    
    transform = transforms.Compose([
                    transforms.Resize(input_size), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                    ])

    if cifar_10 == True:

        train_dataset = CIFAR10(root=train_filepath, train=True, download=True, transform=transform)

        val_dataset = CIFAR10(root=val_filepath, train=False, download=True, transform=transform)

    else:

        # Load the training set from the root folder:
        train_dataset = ImageFolder(root = train_filepath, transform = transform) if train_filepath is not None else []

        # Load the validation set from the root folder:
        val_dataset = ImageFolder(root = val_filepath, transform = transform) if val_filepath is not None else []

    # Create DataLoader objects for both the training and validation sets:  
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers) 
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=num_workers)

    return train_loader, val_loader



def split_data_loader(loader, ratio):

    ratio = torch.as_tensor(ratio)
    train_len = int(len(loader.dataset) * ratio[0])
    val_len = len(loader.dataset) - train_len

    train_set, val_set = random_split(loader.dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=loader.batch_size)
    val_loader = DataLoader(val_set, batch_size=loader.batch_size)
    del train_set, val_set
    
    return train_loader, val_loader




# ========================================================================================
# === MODEL PARALLELIZATION ===
# ========================================================================================

def parallelize_model(model, device_ids):

    if device_ids is not None:
        print("\nParallelizing across {n_gpu} GPUs !!!\n".format(n_gpu=len(device_ids)))
        return nn.DataParallel(model)
    else:
        return model




# ========================================================================================
# === SAVING AND LOADING CHECKPOINTS ===
# ========================================================================================

def save_checkpoint(model, optimizer, epoch, counter_val, counter_train, best_top_k, best_loss, save_path):

    model_class = type(model).__name__ 
    if model_class == 'DataParallel':
        model_to_save = model.module
    else:
        model_to_save = model 

    to_save = {
        'model' : model_to_save,
        'epoch': epoch,
        'counter_val': counter_val,
        'counter_train': counter_train,
        'top_k': best_top_k,
        'loss' : best_loss
    }

    torch.save(to_save, save_path)



def load_checkpoint(model, load_path, device, load_anyways=False, reset_metrics=False, unwrap_data_parallel=False):
 
    checkpoint = torch.load(load_path, map_location=device)
    if unwrap_data_parallel:
        cp_model = checkpoint['model'].module
    else:
        cp_model = checkpoint['model']

    # =========================================
    #           LOADING THE MODEL
    # =========================================
    
    cp_dict = cp_model.state_dict()
    model_dict = model.state_dict()

    # Make sure the classes are the same
    cp_class = type(cp_model).__name__
    model_class = type(model).__name__ 

    if (cp_class != model_class):
        print("\n" + "WARNING: Class mismatch! - Checkpoint model and initialized model aren't instances of the same class!!!")
        print("Checkpoint model class - {}; Initialized model class - {}".format(cp_class, model_class))

        if not load_anyways:
            raise Exception("ERROR: Class mismatch caused program to exit!")
            sys.exit(1)


    # Make sure that state dictionaries of the checkpoint and the initialized model have the same keys 
    cp_keys = set(cp_dict.keys())
    model_keys = set(model_dict.keys())

    if (cp_keys != model_keys):        
        print("\n" + "WARNING: State dict keys mismatch! - Checkpoint model and initialized model don't have the same set of parameters!!!")
        
        if not load_anyways:
            raise Exception("ERROR: Model state dict keys mismatch caused program to exit!")
            sys.exit(1)
    
        print("Trying to load the checkpoint anyways..." + 
        "This makes sense ONLY IF same key names correspond to same functionalities! \n" + 
        "I hope you know what you're doing!")

        if (cp_keys - model_keys):
            print("\n" + "WARNING: Ignoring keys in checkpoint's state dict: {}".format(cp_keys - model_keys))
        if (model_keys - cp_keys):
            print("\n" + "WARNING: Ignoring keys in initialized model's state dict: {}".format(model_keys - cp_keys))
        
        # Take only the shared keys and load them into the model
        dict_to_load = {key: value for key, value in cp_dict.items() if key in model_keys}
        
    else:
        dict_to_load = cp_dict

    # Load the desired parameters
    model.load_state_dict(dict_to_load)    


    # =========================================
    #         LOADING TRACKED METRICKS
    # =========================================
    
    starting_epoch = checkpoint['epoch']
    best_top_k = checkpoint['top_k']
    best_loss = checkpoint['loss']
    counter_val = checkpoint['counter_val']
    counter_train = checkpoint['counter_train']

    if reset_metrics:
        metrics = [0, 0, 0, -1, np.inf]
    else:
        metrics = [starting_epoch, counter_val, counter_train, best_top_k, best_loss]


    return model, metrics

