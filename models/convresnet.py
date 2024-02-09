
import torch.nn as nn
from models.modules import EmbSeq, ConvBlock, FCBlock
import torch

class ConvResNet(nn.Module):
    def __init__(self, settings):
        super(ConvResNet, self).__init__()
        self.blockType = settings.blockType
        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.depth = settings.depth
        self.num_embs = self.depth
        self.num_matrices = 2*self.depth
        self.alpha = settings.alpha
        self.activation = settings.activation
        layers = [(nn.Conv2d(self.num_input_channels, self.width, 2, 2), False),
                  (nn.BatchNorm2d(self.width), False)]

        layers += [(nn.Conv2d(self.width, self.width, 2, 2), False), (nn.BatchNorm2d(self.width), False), (self.activation, False)]

        if self.blockType == "ConvBlock":
            for i in range(self.depth):
                layers += [(ConvBlock(self.width, self.alpha), False),
                           (self.activation, True)]
        else:

            for i in range(self.depth):
                layers += [(FCBlock(self.width, self.alpha), True)]


        self.layers = EmbSeq(layers)

        self.fc = nn.Linear(self.width*8*8, settings.num_output_classes)

        width = self.width*8*8

    def forward(self, x):

        output, embeddings = self.layers(x, [])
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        return output, embeddings

def convresnet(settings):
    return ConvResNet(settings)
