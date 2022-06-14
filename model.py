import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            #     TODO
            #     Conv2d, followed by LeakyReLU
            #     (Conv2d, followed by BatchNorm2d, followed by LeakyReLU) x N, where 2 < N < 8 or so (start with 4?)
        )

        self.classifier = nn.Sequential(
            #     TODO: include activation function like Sigmoid or Linear to perform classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)  # TODO: is this necessary?
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        pass
