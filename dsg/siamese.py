import torch
import torch_geometric as tg
import torch.nn as nn

class SiameseImage(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int,
        h: int,
        w: int,
        node_level: bool = True,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.node_level = node_level
        if not node_level:
            self.aggregation = aggregation

        cnns = [nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1)
        )]
        for _ in range(num_layers - 1):
            cnns.append(nn.Sequential(
                nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1)
            ))
        
        self.cnns = nn.ModuleList(cnns)

        layer_norm = []
        for i in range(num_layers):
            layer_norm.append(nn.LayerNorm((hid_channels, h//(2**(i+1)), w//(2**(i+1)))))
        self.layer_norms = nn.ModuleList(layer_norm)

        self.linear = nn.Sequential(
            nn.Linear(hid_channels * (h // (2 ** num_layers)) * (w // (2 ** num_layers)), hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, hid_channels)
        )

        graph_convs = []
        for _ in range(num_layers):
            graph_convs.append(tg.nn.conv.GCNConv(hid_channels, hid_channels))
        self.graph_convs = nn.ModuleList(graph_convs)

        self.linear2 = nn.Sequential(
            nn.Linear(hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels)
        )

    def forward(self, x, adj):
        # Phase I: independent CNNs
        b, n, d_in, h, w = x.shape
        x = x.view(b*n, d_in, h, w)
        for cnn, layer_norm in zip(self.cnns, self.layer_norms):
            x = cnn(x)
            x = layer_norm(x)
            x = torch.relu(x)
        
        shrink = 2 ** (len(self.cnns))
        x = x.view(b, n, -1, h // shrink, w // shrink)
        x = x.view(b*n, -1)
        x = self.linear(x)

        big_adj = self._tensor_to_block_diag_torch(adj)
        big_adj = tg.utils.dense_to_sparse(big_adj)[0]

        # Phase II: independent graph convolutions
        for graph_conv in self.graph_convs:
            x = graph_conv(x, big_adj)
            x = torch.relu(x)
        
        x = self.linear2(x)
        x = x.view(b, n, -1)

        if not self.node_level:
            x = self._aggregate(x)

        return x

    def _tensor_to_block_diag_torch(self, tensor):
        n, m, _ = tensor.shape
        result = torch.zeros(n*m, n*m, device=tensor.device, dtype=torch.int)
        mask = torch.kron(torch.eye(n), torch.ones(m, m)).bool()
        reshaped_tensor = tensor.reshape(n*m*m).int()
        result[mask] = reshaped_tensor
        return result

    def _aggregate(self, x):
        if self.aggregation == "sum":
            return x.sum(dim=1)
        elif self.aggregation == "mean":
            return x.mean(dim=1)
        elif self.aggregation == "max":
            return x.max(dim=1)
        else:
            raise ValueError("unknown aggregation method")

        