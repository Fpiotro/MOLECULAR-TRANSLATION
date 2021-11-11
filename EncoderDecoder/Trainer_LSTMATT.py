# ====================================================
# Library
# ====================================================
import time
import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.optim
from torch import nn
from tqdm import tqdm
from utils_LSTMATT import *
from models_LSTMATT import *

# ====================================================
# Device
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# ====================================================
# Trainer
# ====================================================
class Trainer_LSTMATT:

    def __init__(self, params):
        # Data parameters
        self.data_name = params['data_name']  # base name shared by data files

        # Training parameters
        self.epochs = params['epochs']  # number of epochs to train for (if early stopping is not triggered)
        self.encoder_lr = params['encoder_lr']  # learning rate for encoder
        self.decoder_lr = params['decoder_lr']  # learning rate for decoder
        self.grad_clip = params['grad_clip']  # clip gradients at an absolute value
        self.fine_tune_encoder = params['fine_tune_encoder']  # fine-tune
        self.print_freq = params['print_freq']  # print training/validation stats every __ batches
        checkpoint = params['checkpoint'] # path to checkpoint, None if none

        # Initialize / load checkpoint
        if checkpoint is False:

            # Model parameters
            emb_dim = params['emb_dim']
            attention_dim = params['attention_dim']
            decoder_dim = params['decoder_dim']
            dropout = params['dropout']
            trained_weigths = params['trained_weights']

            # Training parameters
            self.start_epoch = params['start_epoch']
            self.epochs_since_improvement = params['epochs_since_improvement']  # keeps track of number of epochs since there's been an improvement in validation

            self.best_cross = params['best_cross']  # Cross entropy loss score right now

            # Decoder Initialization
            self.decoder = DecoderWithAttention(attention_dim=attention_dim,
                                                embed_dim=emb_dim,
                                                decoder_dim=decoder_dim,
                                                vocab_size=len(tokenizer),
                                                dropout=dropout,
                                                device=device)

            # Initialization of the decoder optimizer
            self.decoder_optimizer = torch.optim.Adam(params=self.decoder.parameters(),
                                             lr=self.decoder_lr)
            # Encoder Initialization
            self.encoder = Encoder()

            # Loading weights
            if trained_weigths:
                trained_weigths_path = params['trained_weights_path']
                self.encoder.load_weights(trained_weigths_path)

            # Fine tune Encoder
            self.encoder.fine_tune(self.fine_tune_encoder)

            # Initialization of the encoder optimizer
            self.encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                             lr=self.encoder_lr) if self.fine_tune_encoder else None

            # Scheduler for learning rate
            self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=7, gamma=0.1)
            self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=7, gamma=0.1) if self.fine_tune_encoder else None

        else:
            checkpoint_path = params['checkpoint_path']
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.start_epoch = checkpoint['epoch']
            self.epochs_since_improvement = checkpoint['epochs_since_improvement']
            self.best_cross = checkpoint['cross']
            self.encoder = checkpoint['encoder']
            self.encoder_optimizer = checkpoint['encoder_optimizer']
            self.encoder_scheduler = checkpoint['encoder_scheduler']
            
            if self.fine_tune_encoder is True and self.encoder_optimizer is None:
                self.encoder.fine_tune(self.fine_tune_encoder)
                self.encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                                 lr=self.encoder_lr)
                self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=7, gamma=0.1)

            self.decoder = checkpoint['decoder']
            self.decoder_optimizer = checkpoint['decoder_optimizer']
            self.decoder_scheduler = checkpoint['decoder_scheduler']
            
        # Learning rate
        for g in self.decoder_optimizer.param_groups:
            g['lr'] = self.decoder_lr
        if self.fine_tune_encoder:
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = self.encoder_lr

        # Move to GPU, if available
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"]).to(device)


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
            recent_cross = self.val_model(val_loader=val_loader)

            if self.fine_tune_encoder:
                self.encoder_scheduler.step()
            self.decoder_scheduler.step()
                
            # Check if there was an improvement
            is_best = recent_cross > self.best_cross
            self.best_cross = max(recent_cross, self.best_cross)
            if not is_best:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (self.epochs_since_improvement,))
            else:
                self.epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(self.data_name, epoch, self.epochs_since_improvement, self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer, recent_cross, is_best)


    def train_model(self, train_loader, epoch):

        """
        Performs one epoch's training.
        :param train_loader: DataLoader for training data
        :param epoch: epoch number
        """

        # switch to train mode
        self.encoder.train() # train mode (dropout and batchnorm is used)
        self.decoder.train() # train mode (dropout and batchnorm is used)

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        start = time.time()

        for i, (images, labels, label_lengths) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)

            # Batch size
            batch_size = images.size(0)

            # Forward prop.
            features = self.encoder(images)
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(features, labels, label_lengths)
            targets = caps_sorted[:, 1:]
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss.
            loss = self.criterion(predictions, targets)

            # Record loss
            losses.update(loss.item(), batch_size)

            loss.backward()

            # Update weights
            if self.fine_tune_encoder:
                    self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Back prop.
            if self.fine_tune_encoder:
                self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Keep track of metrics
            batch_time.update(time.time() - start)

            # Clip gradients
            #if self.grad_clip is not None:
            #   nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)
            #    if self.encoder_optimizer is not None:
            #        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.encoder.parameters()), self.grad_clip)


            start = time.time()

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.6f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time,                                                                          data_time=data_time, loss=losses))

            #Checkpoint
            if i % 2000 ==0:
              save_checkpoint(self.data_name, epoch, self.epochs_since_improvement, self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer)
              print('\n Model saved \n')

    def val_model(self, val_loader):

        """
        Performs one epoch's validation.
        :param val_loader: DataLoader for validation data.
        """

        # switch to test mode
        self.encoder.eval() # eval mode (no dropout or batchnorm)
        self.decoder.eval() # eval mode (no dropout or batchnorm)

        batch_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()

        # explicitly disable gradient calculation to avoid CUDA memory error
        with torch.no_grad():
            # Batches
            for i, (images, labels, label_lengths) in enumerate(val_loader):

                # Move to GPU, if available
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                # Batch size
                batch_size = images.size(0)

                # Forward prop.
                features = self.encoder(images)
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(features, labels, label_lengths)
                targets = caps_sorted[:, 1:]
                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                # Calculate loss.
                loss = self.criterion(predictions, targets)

                # Record loss
                losses.update(loss.item(), batch_size)
                batch_time.update(time.time() - start)

                start = time.time()

                if i % self.print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.7f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,loss=losses))

        return loss

    def predict(self, image, tokenizer):
        """
        Performs one prediction.
        :param image: torch vector of image.
        """
        self.encoder.eval()# eval mode (no dropout or batchnorm)
        self.decoder.eval()# eval mode (no dropout or batchnorm)
        with torch.no_grad():
            # Move to device, if available
            imgs = image.unsqueeze(0).to(device)
            # Forward prop.
            features = self.encoder(imgs)
            predictions = self.decoder.predict(features, 270, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)

        return _text_preds
