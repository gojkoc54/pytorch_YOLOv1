import argparse
import torchvision
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time 

from util_layers import *
from darknet import Darknet
from utils import *
from trainer import Darknet_Trainer, save_checkpoint

# ========================================================================================
# === GLOBALS ===
# ========================================================================================

parser = argparse.ArgumentParser(description='Darknet training')

parser.add_argument('--train-path', dest='TRAIN_FILEPATH', default='../data/classification/all_together',  
                    type=str, help='Relative path of the training dataset folder')

parser.add_argument('--val-path', dest='VAL_FILEPATH', default=None,  
                    type=str, help='Relative path of the validation dataset folder')

parser.add_argument('--cifar-10', dest='CIFAR_10', type=str2bool , default=False)

parser.add_argument('--num-classes', dest='NUM_CLASSES', default=1000, type=int)  

parser.add_argument('--input-size', dest='YOLO_INPUT_SIZE', default=(448, 448), type=tuple)

parser.add_argument('--fast-yolo', dest='FAST_VERSION', type=str2bool, default=False)

parser.add_argument('--bs', '--batch-size', dest='BATCH_SIZE', default=8, type=int)

parser.add_argument('--nw', '--num-workers', dest='NUM_WORKERS', default=0, type=int)  

parser.add_argument('--epochs', dest='NUM_EPOCHS', default=50, type=int)

parser.add_argument('--val-step', dest='VAL_STEP', default=5000, type=int)

parser.add_argument('--lr', '--learning-rate', dest='LEARNING_RATE', default=1e-3, type=float)

parser.add_argument('--momentum', dest='MOMENTUM', default=0.9, type=float)

parser.add_argument('--wd', '--weight-decay', dest='WEIGHT_DECAY', default=5e-4, type=float)

parser.add_argument('--top-k', dest='TOP_K', default=1, type=int)  

parser.add_argument('--cp-path-topk', dest='CHECKPOINT_PATH_TOP_K', default='../checkpoints/best_checkpoint_top_k.pt', 
                    type=str, help='Relative path to where the model with the best Top-k performance will be stored')

parser.add_argument('--cp-path-loss', dest='CHECKPOINT_PATH_LOSS', default='../checkpoints/best_checkpoint_loss.pt', 
                    type=str, help='Relative path to where the model with the best training performance will be stored')

parser.add_argument('--cp-path-interrupt', dest='INTERRUPT_CHECKPOINT_PATH', default='../checkpoints/interrupt_checkpoint.pt', 
                    type=str, help='Relative path to where the current model will be saved if the program is interrupted')

parser.add_argument('--cp-step', dest='CHECKPOINT_STEP', default=1000, type=int)

parser.add_argument('--load-cp', dest='LOAD_CHECKPOINT', type=str2bool, default=False)

parser.add_argument('--load-conv-cp', dest='LOAD_CONV_LAYERS', type=str2bool, default=False)

parser.add_argument('--load-params-only', dest='LOAD_PARAMS_ONLY', type=str2bool, default=False)

parser.add_argument('--reset-metrics', dest='RESET_METRICS', type=str2bool, default=False)

parser.add_argument('--cp-load-path', dest='CHECKPOINT_LOAD_PATH', default='../checkpoints/interrupt_checkpoint.pt', type=str)

parser.add_argument('--writer-path', dest='WRITER_PATH', default='../runs/ImageNet', type=str)


args = parser.parse_args()



# ========================================================================================
# === MAIN ===
# ========================================================================================


def main():

    # ===== LOAD DATASETS =====

    train_loader, val_loader = load_datasets(train_filepath=args.TRAIN_FILEPATH, val_filepath=args.VAL_FILEPATH, 
                                             input_size=args.YOLO_INPUT_SIZE, batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS)

    if (len(val_loader.dataset) == 0):
        train_loader, val_loader = split_data_loader(train_loader, [0.8, 0.2])


    # ===== DEVICE AND MODEL SETUP =====

    # Device setup:
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if n_gpu else "cpu")
    
    gpu_name = "" if (device == torch.device('cpu')) else torch.cuda.get_device_name(device)
    print("\nThe default device is: {dev} {gpu_name}\n".format(dev=device, gpu_name="; " + gpu_name))
    
    should_parallelize = n_gpu > 1
    device_ids = list(range(n_gpu)) if should_parallelize else None

    # Criterion setup:
    criterion = nn.CrossEntropyLoss().to(device)

    # Tensorboard Summary Writer setup:
    writer = SummaryWriter(args.WRITER_PATH)

    # Initialize the model
    model = Darknet(num_classes = args.NUM_CLASSES, fast_version = args.FAST_VERSION) 


    # ===== LOAD CHECKPOINT =====

    [starting_epoch, counter_val, counter_train, best_top_k, best_loss] = [0, 0, 0, -1, np.inf]

    if (args.LOAD_CHECKPOINT == True):
        cpu = torch.device('cpu')
        model, metrics = load_checkpoint(model, args.CHECKPOINT_LOAD_PATH, device=cpu, load_anyways=True, reset_metrics=args.RESET_METRICS)
        [starting_epoch, counter_val, counter_train, best_top_k, best_loss] = metrics


    # ===== PARALLELIZE MODEL =====    
    
    model = parallelize_model(model, device_ids)
    model = model.to(device)

    # Initialize the optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr = args.LEARNING_RATE, momentum = args.MOMENTUM, weight_decay = args.WEIGHT_DECAY)


    # ===== INITIALIZE TRAINER ====

    trainer = Darknet_Trainer(model, optimizer, criterion, best_top_k, best_loss, counter_val, counter_train, 
                              k = args.TOP_K,
                              num_epochs = args.NUM_EPOCHS, 
                              batch_size = args.BATCH_SIZE,
                              starting_epoch = starting_epoch,
                              checkpoint_path_top_k = args.CHECKPOINT_PATH_TOP_K,
                              checkpoint_path_loss = args.CHECKPOINT_PATH_LOSS,
                              val_step = args.VAL_STEP,
                              checkpoint_step = args.CHECKPOINT_STEP,
                              device = device)

    # ===== START TRAINING ====

    try:
        trainer.train(train_loader, val_loader, writer)

    except (KeyboardInterrupt, SystemExit):

        if (trainer.train_loss_tracker.average < trainer.best_loss):
                    
            trainer.best_loss = trainer.train_loss_tracker.average

        save_checkpoint(trainer.model, trainer.optimizer, trainer.epoch, trainer.val_topk_tracker.mini_batch_num, 
                   trainer.train_loss_tracker.mini_batch_num, trainer.best_top_k, trainer.best_loss, args.INTERRUPT_CHECKPOINT_PATH)




if __name__ == '__main__':
    main()
