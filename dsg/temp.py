import torch
import torch_geometric as tg
from dsg.ultra.datasets import FB15k237Inductive

data = FB15k237Inductive('./data', "v1")

dataloader = tg.loader.DataLoader(data)

for batch in dataloader:
    edge_index = batch.edge_index
    