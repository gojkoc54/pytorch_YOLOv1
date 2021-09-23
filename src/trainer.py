import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys

from utils import save_checkpoint, load_checkpoint


# ========================================================================================
# === METRIC FUNCTIONS AND CLASSES ===
# ========================================================================================

class Metric_tracker():

    def __init__(self):

        self.reset()


    def update(self, new_entry):

        self.mini_batch_num += 1
        self.sum += new_entry
        self.average = self.sum / self.mini_batch_num
        self.latest = new_entry


    def reset(self):

        self.mini_batch_num = 0
        self.sum = 0
        self.average = 0
        self.latest = 0


    def set(self, average, mini_batch_num):

        self.average = average
        self.mini_batch_num = mini_batch_num
        self.sum = average * mini_batch_num
        self.latest = average



def calculate_top_k(output, target, k = 5):

    batch_size = output.shape[0]

    top_k_predictions = torch.topk(output, k)
    top_k_indices = top_k_predictions[1]

    total_matches = 0
    for curr_index, curr_top_k in enumerate(top_k_indices):

        curr_matches = torch.where(curr_top_k == target[curr_index])
        total_matches += curr_matches[0].shape[0]

    return total_matches / batch_size



# ========================================================================================
# === TRAINER CLASSES ===
# ========================================================================================

class Trainer():

    def __init__(self, model, optimizer, criterion, best_top_k, best_loss, counter_val, counter_train, num_epochs, batch_size, k, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.best_top_k = best_top_k
        self.best_loss = best_loss
        self.counter_val = counter_val
        self.counter_train = counter_train
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.k = k

        # Additional training arguments:
        self.starting_epoch = kwargs['starting_epoch'] if 'starting_epoch' in kwargs.keys() else 0
        self.checkpoint_path_top_k = kwargs['checkpoint_path_top_k'] if 'checkpoint_path_top_k' in kwargs.keys() else 'best_checkpoint_top_k.pt'
        self.checkpoint_path_loss = kwargs['checkpoint_path_loss'] if 'checkpoint_path_loss' in kwargs.keys() else 'best_checkpoint_loss.pt'
        self.interrupt_checkpoint_path = kwargs['interrupt_checkpoint_path'] if 'interrupt_checkpoint_path' in kwargs.keys() else 'interrupt_checkpoint.pt'
        self.val_step = kwargs['val_step'] if 'val_step' in kwargs.keys() else None
        self.checkpoint_step = kwargs['checkpoint_step'] if 'checkpoint_step' in kwargs.keys() else self.val_step

        default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = kwargs['device'] if 'device' in kwargs.keys() else default_device




class Darknet_Trainer(Trainer):

    def train(self, train_loader, val_loader, writer):

        print("BEST TOP K: ", self.best_top_k)
        print("BEST LOSS: ", self.best_loss)


        if (self.val_step == None):
            self.val_step = len(train_loader)

        if (self.checkpoint_step == None):
            self.checkpoint_step = self.val_step


        self.val_topk_tracker = Metric_tracker()
        self.train_loss_tracker = Metric_tracker()

        if (self.best_top_k > -1):
            self.val_topk_tracker.set(self.best_top_k, self.counter_val)
        if not (self.best_loss == np.inf):
            self.train_loss_tracker.set(self.best_loss, self.counter_train)

        
        # Training and Validation Loop:
        for self.epoch in range(self.starting_epoch, self.num_epochs):

            self.model.train()

            # Initialize the counter and training statistics:
            i = 0

            for data in train_loader:

                # Load the current mini-batch and move it to 'device':
                input, target = data[0].to(self.device), data[1].to(self.device)

                # Reset the optimizer gradient:
                self.optimizer.zero_grad()

                # Forward pass:
                output = self.model(input)

                # Loss:
                loss = self.criterion(output, target)
                
                # Backward pass and optimize:
                loss.backward()
                self.optimizer.step()

                # Training statistics
                self.train_loss_tracker.update(float(loss))

                if (i % 10 == 9):
                    print('[%d, %5d] loss: %.3f' %
                        (self.epoch + 1, i + 1, float(loss)))
                
                # Tensorboard:
                writer.add_scalar('Training loss mean', self.train_loss_tracker.average, self.train_loss_tracker.mini_batch_num)
                writer.add_scalar('Training loss', self.train_loss_tracker.latest, self.train_loss_tracker.mini_batch_num)

                i += 1



                # If CHECKPOINT_STEP mini-batches have passed through the training loop ==>> Checkpoint?:
                if (i % self.checkpoint_step == self.checkpoint_step - 1):

                    # Save the current model if it's better than the previous checkpoint:
                    if (self.val_topk_tracker.average > self.best_top_k):

                        self.best_top_k = float(self.val_topk_tracker.average)

                        save_checkpoint(self.model, self.optimizer, self.epoch, self.val_topk_tracker.mini_batch_num, self.train_loss_tracker.mini_batch_num, self.best_top_k, self.best_loss, save_path = self.checkpoint_path_top_k)

                    if (self.train_loss_tracker.average < self.best_loss):

                        self.best_loss = self.train_loss_tracker.average

                        save_checkpoint(self.model, self.optimizer, self.epoch, self.val_topk_tracker.mini_batch_num, self.train_loss_tracker.mini_batch_num, self.best_top_k, self.best_loss, save_path = self.checkpoint_path_loss)


                # If VAL_STEP mini-batches have passed through the training loop ==>> Validation:
                if not (i % self.val_step == self.val_step - 1):
                    continue

                self.model.eval()

                i_val = 0
                for data in val_loader:
                    
                    input, target = data[0].to(self.device), data[1].to(self.device)

                    # Forward pass
                    output = self.model(input)

                    # Validation statistics
                    top_k = calculate_top_k(output, target, k = self.k)
                    self.val_topk_tracker.update(top_k)
                    
                    if (i_val % 10 == 9):
                        print('[%d, %5d] VAL' % 
                            (self.epoch + 1, i_val + 1))

                    # Tensorboard:
                    writer.add_scalar('Top-{} accuracy'.format(str(self.k)), float(self.val_topk_tracker.average), self.val_topk_tracker.mini_batch_num)
                    writer.add_scalar('Top-{} accuracy per mini-batch'.format(str(self.k)), float(self.val_topk_tracker.latest), self.val_topk_tracker.mini_batch_num)
                    i_val += 1

                    del input, target, data

                
        writer.close()