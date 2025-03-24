import torch
import torch_geometric as tg
import torch.nn as nn

class SiameseImage(nn.Module):

    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        num_layers,
    ):
        super().__init__()
        