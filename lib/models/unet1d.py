import torch
import torch.nn as nn
import torch.nn.functional as F

'''Adapted from https://github.com/milesial/Pytorch-UNet by James Howard (2D -> 1D)'''


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=None):
        super(Down, self).__init__()
        self.dropout = dropout
        self.mpconv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        if self.dropout:
            x = F.dropout(x, self.dropout)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diff = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, (diff // 2, diff - diff // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=6, starting_layers=64, dropout=None):
        super(UNet1D, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.inc = InConv(in_channels, starting_layers)
        self.down1 = Down(starting_layers * 1, starting_layers * 2, dropout)  # Only dropout on early layers
        self.down2 = Down(starting_layers * 2, starting_layers * 4, dropout)
        self.down3 = Down(starting_layers * 4, starting_layers * 8, dropout)
        self.down4 = Down(starting_layers * 8, starting_layers * 8, dropout)
        if self.n_layers >= 5:
            self.down5 = Down(starting_layers * 8, starting_layers * 8)
            if self.n_layers >= 6:
                self.down6 = Down(starting_layers * 8, starting_layers * 8)
                self.up6 = Up(starting_layers * 16, starting_layers * 8)
            self.up5 = Up(starting_layers * 16, starting_layers * 8)
        self.up4 = Up(starting_layers * 16, starting_layers * 4)
        self.up3 = Up(starting_layers * 8, starting_layers * 2)
        self.up2 = Up(starting_layers * 4, starting_layers)
        self.up1 = Up(starting_layers * 2, starting_layers)
        self.out = OutConv(starting_layers, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.n_layers >= 4:
            x5 = self.down4(x4)
            x = x5
            if self.n_layers >= 5:
                x6 = self.down5(x5)
                x = x6
                if self.n_layers >= 6:
                    x7 = self.down6(x6)
                    x = x7
                    x = self.up6(x, x6)
                x = self.up5(x, x5)
            x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)
        return x
