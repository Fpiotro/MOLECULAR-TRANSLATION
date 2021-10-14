# ====================================================
# Library
# ====================================================
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from tqdm import tqdm
from utils_AE import *
from models_AE import *

# ====================================================
# Device
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# ====================================================
# Trainer
# ====================================================
class Trainer:

    def __init__(self, params):
        # Data parameters
        self.data_name = params['data_name']  # base name shared by data files

        # Training parameters
        self.epochs = params['epochs']  # number of epochs to train for (if early stopping is not triggered)
        self.model_lr = params['model_lr']  # learning rate for model
        self.grad_clip = params['grad_clip']  # clip gradients at an absolute value
        self.print_freq = params['print_freq']  # print training/validation stats every __ batches
        checkpoint = params['checkpoint'] # path to checkpoint, None if none

        # Initialize / load checkpoint
        if checkpoint is False:

            # Model parameters
            n_channels = params['n_channels']
            output_dim = params['output_dim']
            
            # Training parameters
            self.start_epoch = params['start_epoch']
            self.epochs_since_improvement = params['epochs_since_improvement']  # keeps track of number of epochs since there's been an improvement in validation

            self.best_mse = params['best_mse']  # MSE score right now

            self.model = UNet(n_channels, output_dim)
            self.model_optimizer = torch.optim.Adam(self.model.parameters(),lr = self.model_lr)

            # Scheduler for learning rate
            self.model_scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=7, gamma=0.1)

        else:
            checkpoint_path = params['checkpoint_path']
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.start_epoch = checkpoint['epoch']
            self.epochs_since_improvement = checkpoint['epochs_since_improvement']
            self.best_mse = checkpoint['mse']
            self.model = checkpoint['model']
            self.model_optimizer = checkpoint['model_optimizer']
            self.model_scheduler = checkpoint['model_scheduler']

        # Learning rate
        for g in self.model_optimizer.param_groups:
            g['lr'] = self.model_lr
            
        # Move to GPU, if available
        self.model = self.model.to(device)

        # Loss function
        self.criterion = nn.MSELoss().to(device)

    def train_val_model(self, train_loader, val_loader):

        """
        Performs epochs training and validation.
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data.
        """

        # Epochs
        for epoch in range(self.start_epoch, self.epochs):

            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if self.epochs_since_improvement == 20:
                break

            # One epoch's training
            self.train_model(train_loader=train_loader, epoch=epoch)

            # One epoch's validation
            recent_mse = self.val_model(val_loader=val_loader)

            self.model_scheduler.step()

            # Check if there was an improvement
            is_best = recent_mse > self.best_mse
            self.best_mse = max(recent_mse, self.best_mse)
            if not is_best:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (self.epochs_since_improvement,))
            else:
                self.epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(self.data_name, epoch, self.epochs_since_improvement, self.model, self.model_optimizer, self.model_scheduler, recent_mse, is_best)


    def train_model(self, train_loader, epoch):

        """
        Performs one epoch's training.
        :param train_loader: DataLoader for training data
        :param epoch: epoch number
        """

        # switch to train mode
        self.model.train() # train mode (dropout and batchnorm is used)

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        start = time.time()

        for i, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Gradient zero.
            self.model_optimizer.zero_grad()

            # Forward prop.
            predictions = self.model(images)

            # Calculate loss.
            loss = self.criterion(predictions.float(), labels.float())

            # Back prop.
            loss.backward()

            # Clip gradients
            if self.grad_clip is not False:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.grad_clip)

            # Update weights
            self.model_optimizer.step()

            # Keep track of metrics
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.6f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time,                                                                          data_time=data_time, loss=losses))

            #Checkpoint
            if i % 2000 ==0:
              save_checkpoint(self.data_name, epoch, self.epochs_since_improvement, self.model, self.model_optimizer, self.model_scheduler)
              print('\n Model saved \n')

    def val_model(self, val_loader):

        """
        Performs one epoch's validation.
        :param val_loader: DataLoader for validation data.
        """

        self.model.eval()  # eval mode (no dropout or batchnorm)

        batch_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()

        # explicitly disable gradient calculation to avoid CUDA memory error
        with torch.no_grad():
            # Batches
            for i, (imgs, labs) in enumerate(val_loader):

                # Move to device, if available
                imgs = imgs.to(device, non_blocking=True)
                labs = labs.to(device, non_blocking=True)

                # Forward prop.
                scores = self.model(imgs)

                # Calculate loss
                loss = self.criterion(scores.float(), labs.float())

                # Keep track of metrics
                losses.update(loss.item(), imgs.size(0))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % self.print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,loss=losses))

        return loss
    
    def predict(self, image, numpy=False):
        """
        Performs one prediction.
        :param image: Tensor.
        :param numpy: return a numpy array or tensor
        """
        self.model.eval()
        pred = self.model(image.unsqueeze(0).to(device)).detach().cpu().squeeze()
        if numpy:
            pred = (pred.permute(1,2,0).numpy()*255).astype(np.uint8)
        return pred
