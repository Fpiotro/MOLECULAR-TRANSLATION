import torch
from torch import nn
import torchvision

class DLChI(nn.Module):
    """
    Deep Learning model for InChI.
    """

    def __init__(self, output_dim, encoded_image_size=4, dropout=0.1):

        super(DLChI, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.seq = nn.Sequential(nn.Flatten(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(2048*encoded_image_size*encoded_image_size, 1000, bias = True),
                                 nn.Sigmoid(),
                                 nn.Linear(1000, output_dim, bias = True),
                                 nn.Sigmoid())

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: tensor with dim (nb atoms)
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = self.seq(out)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        if fine_tune:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune