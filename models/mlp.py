
import torch.nn as nn
from models.modules import EmbSeq
import torch

class MLP(nn.Module):

    def __init__(self, settings):
        super().__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.num_embs = self.num_matrices = self.depth = settings.depth
        self.activation = settings.activation
        self.bn = settings.bn

        layers = []

        width = self.num_input_channels*32*32
        for i in range(self.depth):
            layers += [(nn.Linear(width, self.width), True)]
            if self.bn: layers += [(nn.BatchNorm1d(self.width), False)]
            layers += [(self.activation, False)]
            width = self.width

        self.layers = EmbSeq(layers)

        self.fc = nn.Linear(width, settings.num_output_classes)

    def forward(self, x):

        output = x.view(x.shape[0], -1)
        output, embeddings = self.layers(output, [])
        output = self.fc(output)

        return output, embeddings


def mlp(settings):
    return MLP(settings)
