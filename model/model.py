import torch
import torch.nn as nn


# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py

# TODO: add cube sphere padding before each convolutional layer

class ResidualConvBlock(nn.Module):
    """Implements residual conv function.
    Args:
        channels (int): Number of channels in the input.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (128) x 20 x 20
            nn.Conv2d(128, 64, (3, 3), (1, 1), (0, 0), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 18 x 18
            nn.Conv2d(64, 64, (3, 3), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 16 x 16
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 8 x 8
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # First conv layer.
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(128, 1024, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(1024))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(1024),
        )

        # Upscale block
        upsampling = []
        for _ in range(2):
            upsampling.append(UpsampleBlock(1024))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv_block3 = nn.Conv2d(1024, 128, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)

        out = torch.clamp_(out, -3.0, 3.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
