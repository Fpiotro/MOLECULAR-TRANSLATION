# ====================================================
# Library
# ====================================================
import numpy as np
import random
import os
import torch
import matplotlib.pyplot as plt
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import MolDrawOptions
from io import BytesIO
import lxml.etree as et
from PIL import Image
import cv2
import cairosvg
import pathlib
from configparser import ConfigParser

# ====================================================
# Path
# ====================================================
BASE_PATH = pathlib.Path(__file__).parent.resolve()
INI_PATH = BASE_PATH.joinpath("ini").resolve()

# ====================================================
# params.ini
# ====================================================
config = ConfigParser()
config.read(INI_PATH.joinpath("params.ini"))
output_path = config['Data_parameters']['data_path_output']

# ====================================================
# Path
# ====================================================
CHECK_PATH = pathlib.Path(output_path).joinpath("checkpoint").resolve()

if not os.path.isdir(CHECK_PATH):
    os.mkdir(CHECK_PATH)

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

    filename = 'checkpoint_' + data_name + '_' + str(epoch) + '.pth.tar'
    torch.save(state, CHECK_PATH.joinpath(filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, directory + 'BEST_' + filename)
        
# read International Chemical Identifier
def draw_mol(fb, size1, size2):
    """
    Returns image of InchI.
    :param fb: InchI
    :param size1: first size of the image
    :param size2: second size of the image
    """
    mol = Chem.inchi.MolFromInchi(fb)
    # draw molecule with angle degree rotation
    d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(size1, size2)
    AllChem.Compute2DCoords(mol)
    d.drawOptions().bondLineWidth = 1
    d.drawOptions().useDefaultAtomPalette()
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText("0.png")
    return (255- cv2.imread("0.png", cv2.IMREAD_UNCHANGED))/255

def svg_to_image(svg):
    """
    Returns image from svg.
    :param svg: object in svg
    """
    svg_str = et.tostring(svg)
    # TODO: would prefer to convert SVG dirrectly to a numpy array.
    png = cairosvg.svg2png(bytestring=svg_str)
    image = np.array(Image.open(BytesIO(png)), dtype=np.uint8)
    return image

def random_molecule_image(inchi, render_size1, render_size2, drop_bonds=True):
    """
    Returns image of InchI with drop bonds.
    :param inchi: InchI
    :param render_size1: first size of the image
    :param render_size2: second size of the image
    """
    # Note that the original image is returned as two layers: one for atoms and one for bonds.
    #mol = Chem.MolFromSmiles(smiles)
    mol = Chem.inchi.MolFromInchi(inchi)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(render_size1, render_size2)
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.bondLineWidth = 1
    d.SetDrawOptions(options)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg_str = d.GetDrawingText()
    # Do some SVG manipulation
    svg = et.fromstring(svg_str.encode('iso-8859-1'))
    atom_elems = svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    bond_elems = svg.xpath(r'//svg:path[starts-with(@class,"bond-")]', namespaces={'svg': 'http://www.w3.org/2000/svg'})

    if drop_bonds:
        num_bond_elems = len(bond_elems)
        nb = np.around(num_bond_elems/10).astype(int)
        for i in range(nb):
          # drop a bond
          # Let's leave at least one bond!
          if num_bond_elems > 1:
            bond_elem_idx = np.random.randint(1,num_bond_elems-1)
            bond_elem = bond_elems[bond_elem_idx]
            bond_parent_elem = bond_elem.getparent()
            if bond_parent_elem is not None:
              bond_parent_elem.remove(bond_elem)
              num_bond_elems -= 1
            else:
              break
    img = svg_to_image(svg)
    return img

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
