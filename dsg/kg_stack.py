import torch
import torch_geometric as tg
import torch.nn as nn

from dsg.ultra.datasets import FB15k237Inductive, WN18RRInductive, NELLInductive, FB15k237_10

class GraphStack(tg.data.Dataset):

    def __init__(self, data, node_features):
        self.data = data
        self.node_features = node_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        return self.transform_graph(datapoint)

    def transform_graph(self, datapoint):
        edge_index = datapoint.edge_index
        edge_type = datapoint.edge_type
        num_relations = datapoint.num_relations
        num_nodes = datapoint.num_nodes

        # split by relations
        sorted_adj = edge_index[:, edge_type]
        uniq, row_counts = torch.unique(edge_type, return_counts=True)
        individual_adj_matrices = torch.split(sorted_adj, row_counts.tolist(), dim=1)


        # adjust num_relations based on actual occurance
        num_relations = len(uniq)
        datapoint.num_relations = num_relations

        # make num_relations graphs with the individual adjacencies
        adjs = []
        unique = []
        for i in range(num_relations):
            adjs.append(individual_adj_matrices[i] + (num_nodes * i))
            unique.append(individual_adj_matrices[i].reshape(-1).unique())

        # merge back into one big graph
        big_adj = torch.cat(adjs, dim=1)

        # compute connections between relations
        binary_matrix = torch.zeros((num_relations, num_nodes), dtype=torch.int32)
        for i, indices in enumerate(unique):
            binary_matrix[i, indices] = 1.
        overlap_matrix = binary_matrix @ binary_matrix.T

        overlap_matrix = connect_relations(overlap_matrix)
        
        x = torch.ones((num_relations, num_nodes, self.node_features))
        datapoint.x = x
        datapoint.sub_adj = big_adj
        datapoint.adj = overlap_matrix
        return datapoint
    
def connect_relations(overlap):
    # connection by absolute tao
    tao = torch.quantile(overlap.float(), 0.8)
    overlap[overlap < tao] = 0
    overlap[overlap >= tao] = 1
    
    return overlap


def get_dataloader(train, node_features):
    data = FB15k237Inductive('./data', "v1")
    data = GraphStack(data, node_features)
    return tg.loader.DataLoader(data)


if __name__ == '__main__':

    data = get_dataloader(True, 4)

    from dsg.DSG import DSGraphG

    model = DSGraphG(4, 128, 10, 3, 1, 4)
    
    from dsg.train import graph_train as train
    from dsg.train import graph_eval as eval

    model = train(model, data, 10)
    eval(model, data)