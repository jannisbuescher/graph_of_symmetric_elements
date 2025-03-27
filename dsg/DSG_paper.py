import torch
import torch.nn as nn
import torch_geometric as tg
import torch.functional as F

class DSImageGLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_agg_fn: bool = False,
                 use_LH2: bool = False,
                 use_LH4: bool = False,
                 sum_agg: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_agg_fn = use_agg_fn
        self.use_LH2 = use_LH2
        self.use_LH4 = use_LH4
        self.sum_agg = sum_agg

        agg_dim = 2
        
        self.LH1 =  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.LH3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        if use_LH2:
            self.LH2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            agg_dim += 1
        
        if use_LH4:
            self.LH4 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            agg_dim += 1
        
        self.agg = nn.Conv2d(agg_dim * out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        
    def forward(self, x, adj):
        b, n, d, h, w = x.shape
        x1 = x.view(b*n, d, h, w)
        # calculate LH1
        LH1_x = self.LH1(x1).view(b, n, -1, h//2, w//2)

        # calculate LH3
        x3 = x
        if adj.is_sparse:
            raise NotImplementedError()
        else:
            # (b*n, d, h, w) -> (b, n, d, h, w), X_i -> \sum A_{ij}x_j
            all_neigh = torch.einsum('bnm,bmdhw->bndhw', adj, x3)
            x3 = all_neigh.view(b*n, d, h, w) # (b*n, d, h, w)
            LH3_x = self.LH3(x3) # (b*n, d', h, w)
            LH3_x = LH3_x.view(b, n, -1, h//2, w//2)

        # calculate LH2
        if self.use_LH2:
            sum_x = x.view(b, n, -1, h, w)
            if self.sum_agg:
                sum_x = sum_x.sum(dim=1)
            else:
                sum_x = sum_x.mean(dim=1) # (b, d, h, w)
            LH2_x = self.LH2(sum_x) # (b, d', h//2, w//2)
            LH2_x = LH2_x.view(b, 1, -1, h//2, w//2).repeat(1, n, 1, 1, 1) # (b, n, d', h//2, w//2)

        # calculate LH4
        if self.use_LH4:
            sum_neigh_x = all_neigh
            if self.sum_agg:
                sum_neigh_x = sum_neigh_x.sum(dim=1)
            else:
                sum_neigh_x = sum_neigh_x.mean(dim=1) # (b, d, h, w)
            LH4_x = self.LH4(sum_neigh_x) # (b, d', h//2, w//2)
            LH4_x = LH4_x.view(b, 1, -1, h//2, w//2).repeat(1, n, 1, 1, 1) # (b, n, d', h//2, w//2)

        if self.use_agg_fn:
            LHx = torch.cat([LH1_x, LH3_x], dim=2) # (b, n, 2d', h//2, w//2)
            if self.use_LH2:
                LHx = torch.cat([LHx, LH2_x], dim=2) # (b, n, 3d, h//2, w//2)
            if self.use_LH4:
                LHx = torch.cat([LHx, LH4_x], dim=2) # (b, n, 4d', h//2, w//2)
            
            LHx = LHx.view(b*n, -1, h//2, w//2)
            LHx = self.agg(LHx) # (b*n, d', h//2, w//2)
            LHx = LHx.view(b, n, -1, h//2, w//2) # (b, n, d', h//2, w//)

        else:
            LHx = LH1_x + LH3_x
            if self.use_LH2:
                LHx += LH2_x
            if self.use_LH4:
                LHx += LH4_x
            
        return LHx
    

class DSImageG(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int,
        h: int,
        w: int,
        use_agg_fn: bool = False,
        use_LH2: bool = False,
        use_LH4: bool = False,
        sum_agg: bool = False,
        node_level: bool = True,
        aggregation: str = 'sum',
    ):
        super().__init__()
        self.out_channels = out_channels
        self.node_level = node_level
        self.aggregation = aggregation

        layers = [DSImageGLayer(in_channels, hid_channels, 
                                use_agg_fn=use_agg_fn, use_LH2=use_LH2, use_LH4=use_LH4, sum_agg=sum_agg)]
        
        for _ in range(num_layers - 1):
            layers.append(DSImageGLayer(hid_channels, hid_channels, 
                                        use_agg_fn=use_agg_fn, use_LH2=use_LH2, use_LH4=use_LH4, sum_agg=sum_agg))

        self.layers = nn.ModuleList(layers)

        layer_norm = []
        for i in range(num_layers):
            layer_norm.append(nn.LayerNorm((hid_channels, h//(2**(i+1)), w//(2**(i+1)))))
        self.layer_norms = nn.ModuleList(layer_norm)

        self.linear = nn.Sequential(
            nn.Linear(hid_channels * (h // (2 ** num_layers)) * (w // (2 ** num_layers)), hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels)
        )

    def forward(self, x, adj):
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            x = layer(x, adj)
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
    
    