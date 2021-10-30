from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: Union[int, tuple, list],
                 stride: Union[int, tuple, list],
                 padding: Union[int, tuple, list],
                 last: bool) -> None:
        super(GeneratorBlock, self).__init__()
        if not last:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
                nn.Tanh()
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_out = self.model(x)
        return x_out


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: Union[int, tuple, list],
                 stride: Union[int, tuple, list],
                 padding: Union[int, tuple, list],
                 last: bool) -> None:
        super(DiscriminatorBlock, self).__init__()
        if not last:
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2)
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x) -> torch.tensor:
        x_out = self.model(x)
        return x_out


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int) -> None:
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            GeneratorBlock(nz, ngf * 8, 4, 1, 0, False),
            GeneratorBlock(ngf * 8, ngf * 4, 4, 2, 1, False),
            GeneratorBlock(ngf * 4, ngf * 2, 4, 2, 1, False),
            GeneratorBlock(ngf * 2, ngf, 4, 2, 1, False),
            GeneratorBlock(ngf, nc, 4, 2, 1, True)
        )

    def forward(self, x) -> torch.tensor:
        x_out = self.model(x)
        return x_out


class Discriminator(nn.Module):
    def __init__(self, nc: int, ndf: int) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            GeneratorBlock(nc, ndf, 4, 2, 1, False),
            GeneratorBlock(ndf, ndf * 2, 4, 2, 1, False),
            GeneratorBlock(ndf * 2, ndf * 4, 4, 2, 1, False),
            GeneratorBlock(ndf * 4, ndf * 8, 4, 2, 1, False),
            GeneratorBlock(ndf * 8, 1, 4, 1, 0, True)
        )

    def forward(self, x):
        x_out = self.model(x)
        return x_out

# test
if __name__ == "__main__":
    G = Generator(100, 64, 3)
    print(G)
    D = Discriminator()
