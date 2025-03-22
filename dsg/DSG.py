import torch
import torch.nn as nn
import torch_geometric as tg
import torch.functional as F

class DSImageGLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        
        L = []
        for _ in range(self.k + 1):
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.L = nn.ModuleList(L)
        
    def forward(self, x, cached_adj):
        b, n, d, h, w = x.shape
        x = x.view(b*n, d, h, w)
        L0_x = self.L[0](x)
        Lx = L0_x.view(b, n, -1, h, w)
        for L, A_i in zip(self.L, cached_adj):
            # (b*n, d, h, w) -> (b, n, d, h, w), X_i -> \sum A_{ij}x_j
            x_i = x.view(b, n, d, h, w)
            if A_i.is_sparse: 
                x_i = torch.sparse.mm(A_i, x_i)
                raise NotImplementedError()
            else:
                x_i = torch.einsum('bnm,bmdhw->bndhw', A_i, x_i)
            x_i = x_i.view(b*n, d, h, w) # (b*n, d, h, w)
            Li_x = L(x_i) # (b*n, d', h, w)
            Lx += Li_x.view(b, n, -1, h, w)
        return Lx
    

class DSImageG(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int,
        k: int,
        h: int,
        w: int,
        node_level: bool = True,
        aggregation: str = 'sum',
    ):
        super().__init__()
        self.out_channels = out_channels
        self.k = k
        self.node_level = node_level
        self.aggregation = aggregation
        hw = h*w

        layers = [DSImageGLayer(in_channels, hid_channels, k)]
        
        for _ in range(num_layers - 1):
            layers.append(DSImageGLayer(hid_channels, hid_channels, k))

        self.layers = nn.ModuleList(layers)

        layer_norm = []
        for _ in range(num_layers):
            layer_norm.append(nn.LayerNorm((hid_channels, h, w)))
        self.layer_norms = nn.ModuleList(layer_norm)

        self.linear = nn.Linear(hid_channels * hw, out_channels)

    def forward(self, x, adj):
        cached_adj = self.precompute_adj(adj)
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            x = layer(x, cached_adj)
            x = layer_norm(x)
            x = torch.relu(x)

        b, n, d, h, w = x.shape
        x = x.view(b * n, d*h*w)
        x = self.linear(x) # (b*n, out_dim)
        x = x.view(b, n, self.out_channels)

        if not self.node_level:
            x = self._aggregate(x)
        return x
    
    def _aggregate(self, x):
        if self.aggregation == 'sum':
            x = x.sum(dim=1)
        elif self.aggregation == 'mean':
            x = x.mean(dim=1)
        elif self.aggregation == 'max':
            x = x.max(dim=1)
        else:
            raise ValueError(f'Invalid aggregation method: {self.aggregation}')
        return x


    def precompute_adj(self, adj):
        cached_adj = [adj.float()]
        A_i = adj
        for _ in range(self.k - 1):
            if A_i.is_sparse:
                A_iplus1 = torch.sparse.mm(A_i, adj) # broken
                raise NotImplementedError()
            else:
                A_iplus1 = torch.einsum('bnm,bmk->bnk', A_i, adj)
            cached_adj.append(A_iplus1.detach().clone().float())
        return cached_adj
    

class DSGraphGLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int
    ):
        self.k = k

        L = [tg.nn.conv.GCNConv(in_channels, out_channels)]

        for _ in range(k):
            L.append(tg.nn.conv.GCNConv(in_channels, out_channels))

        self.L = nn.ModuleList(L)

    def forward(self, x, adj, cached_adj):
        # x: (n, m, d)
        # adj: (m, m)
        n, m, d = x.shape
        L0_x = self.L[0](x, adj)
        Lx = L0_x
        for L, A_i in zip(self.L, cached_adj):
            # (n, m, d) -> (n, m, d), X_i -> \sum A_{ij}x_j
            if A_i.is_sparse: 
                x_i = torch.sparse.mm(A_i, x_i)
                raise NotImplementedError()
            else:
                x_i = torch.einsum('nj,jmd->nmd', A_i, x_i)
            x_i = x_i           # (n, m, d)
            Li_x = L(x_i, adj)  # (n, m, d')
            Lx += Li_x
        return Lx

    
class DSGraphG(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int,
        k: int,
        node_features: int,
        node_level: bool = True,
        aggregation: str = 'sum'
    ):
        super().__init__()
        self.k = k
        self.node_level = node_level
        if node_level:
            self.aggregation = aggregation
        
        layers = [DSGraphGLayer(in_channels, hid_channels, k)]
        
        for _ in range(num_layers - 1):
            layers.append(DSGraphGLayer(hid_channels, hid_channels, k))

        layers.append(DSGraphGLayer(hid_channels, out_channels, k))

        self.layers = nn.ModuleList(layers)

        layer_norm = []
        for _ in range(num_layers):
            layer_norm.append(nn.LayerNorm((hid_channels, node_features)))
        self.layer_norms = nn.ModuleList(layer_norm)

        self.linear = nn.Linear(hid_channels * node_features, out_channels)

    def forward(self, x, sub_adj, adj):
        """x: feature vector, adj: adjacency of meta graph, sub_adj: adjacency of graphs at the node"""
        cached_adj = self.precompute_adj(adj)
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            x = layer(x, sub_adj, cached_adj)
            x = layer_norm(x)
            x = torch.relu(x)

        n, m, d = x.shape
        x = x.view(n*m, d)
        x = self.linear(x) # (n*m, out_dim)
        x = x.view(n, m, self.out_channels)

        if not self.node_level:
            x = self._aggregate(x)
        return x

    def _aggregate(self, x):
        if self.aggregation == 'sum':
            x = x.sum(dim=0)
        elif self.aggregation == 'mean':
            x = x.mean(dim=0)
        elif self.aggregation == 'max':
            x = x.max(dim=0)
        else:
            raise ValueError(f'Invalid aggregation method: {self.aggregation}')
        return x

    def precompute_adj(self, adj):
        cached_adj = [adj.float()]
        A_i = adj
        for _ in range(self.k - 1):
            if A_i.is_sparse:
                A_iplus1 = torch.sparse.mm(A_i, adj) # broken
                raise NotImplementedError()
            else:
                A_iplus1 = torch.einsum('bnm,bmk->bnk', A_i, adj)
            cached_adj.append(A_iplus1.detach().clone().float())
        return cached_adj