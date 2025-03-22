import torch
import torch_geometric as tg
import torch.nn as nn

from dsg.ultra.datasets import FB15k237Inductive

class GraphStack(tg.data.Dataset):

    def __init__(self, data, node_features):
        self.data = data
        self.node_features = node_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        num_nodes = datapoint.num_nodes
        x = torch.ones((num_nodes, self.node_features))
        datapoint.x = x
        return datapoint
    

def transform_graph(datapoint):
    edge_type = datapoint.edge_type
    num_relations = datapoint.num_relations

if __name__ == '__main__':

    data = FB15k237Inductive('./data', "v1")
    data = GraphStack(data)
