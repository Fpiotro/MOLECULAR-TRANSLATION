# ====================================================
# Library
# ====================================================
import numpy as np
import random
import os
import torch
import Levenshtein
import matplotlib.pyplot as plt
import pickle

# ====================================================
# Utils
# ====================================================
def seed_torch(seed=42):
    """
    Defines a random seed.
    :param seed: number of random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(data_name, epoch, epochs_since_improvement, model, model_optimizer, model_scheduler, mse = 10, is_best=False):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in MSE score
    :param model: model
    :param model_optimizer: optimizer to update encoder's weights
    :param MSE: validation MSE score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'mse': mse,
             'model': model,
             'model_optimizer': model_optimizer,
             'model_scheduler': model_scheduler
             }

    directory = './'
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, directory + filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, directory + 'BEST_' + filename)

def save_obj(obj, name):
    """
    Saves object with pickle.
    :param obj: object to be saved
    :param name: name of object in the depository
    """
    with open('F://bms-molecular-translation//'+'obj//'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Loads object with pickle.
    :param name: name of object to be loaded
    """
    with open('F://bms-molecular-translation//'+'obj//' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_score(y_true, y_pred):
    """
    Gets the Levenshtein distance between two strings.
    :param y_true: true label
    :param y_pred: prediction
    """
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score

def mol_fb(fb, atom, norm=True):
    """
    Gives raw formula.
    :param fb: vector with number of atoms present (normalized or not)
    :param atom: dictionary of atoms
    :param norm: vector normalized or not
    """
    if norm:
        max = torch.load("F://bms-molecular-translation//obj//data_train_max.pth")
        min = torch.load("F://bms-molecular-translation//obj//data_train_min.pth")
        fb = (max-min)*fb + min
    fb = np.around(fb.numpy()).astype(int)
    atom = np.array(list(atom.keys()))
    mask = (fb!=0)
    atom = atom[mask]
    fb = fb[mask].astype(str)
    result = np.vstack((atom, fb))
    result = result.flatten('F')
    # result = [None]*(2*len(atom))
    # result[::2] = atom
    # result[1::2] = fb
    return "".join(list(result))

def draw_mol(img, fb=None, title=None):
    """
    Draws molecule with its raw formula (optional).
    :param img: torch vector of image
    :param fb: raw formula (str)
    :param title: title of the plot (str)
    """
    img = img.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std*img + mean
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(img)
    if title is not None:
        plt.suptitle(title)
    plt.show()

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def clip_gradient(optimizer, grad_clip):
#     """
#     Clips gradients computed during backpropagation to avoid explosion of gradients.
#     :param optimizer: optimizer with the gradients to be clipped
#     :param grad_clip: clip value
#     """
#     for group in optimizer.param_groups:
#         for param in group['params']:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)
#
# def adjust_learning_rate(optimizer, shrink_factor):
#     """
#     Shrinks learning rate by a specified factor.
#     :param optimizer: optimizer whose learning rate must be shrunk.
#     :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
#     """
#
#     print("\nDECAYING learning rate.")
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = param_group['lr'] * shrink_factor
#     print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
