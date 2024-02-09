
import torch.nn as nn
from models.modules import EmbSeq
import torch

class ConvNet(nn.Module):
    def __init__(self, settings):
        super(ConvNet, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.num_embs = self.num_matrices = self.depth = settings.depth
        self.activation = settings.activation

        layers = []

        layers += [(nn.Conv2d(self.num_input_channels, self.width, 2, 2), False),
                   (nn.BatchNorm2d(self.width), False)]

        layers += [(nn.Conv2d(self.width, self.width, 2, 2), False),
                   (nn.BatchNorm2d(self.width), False),
                   (self.activation, False)]

        for i in range(self.depth):
            layers += [(nn.Conv2d(self.width, self.width, 3, 1, 1), True),
             (nn.BatchNorm2d(self.width), False),
             (self.activation, False)]

        self.layers = EmbSeq(layers)

        self.fc = nn.Linear(self.width*8*8, settings.num_output_classes)#8

        width = self.width*8*8#8 24

    def forward(self, x):

        output, embeddings = self.layers(x, [])
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        return output, embeddings

def convnet(settings):
    return ConvNet(settings)