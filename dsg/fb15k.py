import torch_geometric as tg
import torch

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphStackFB15K(tg.data.Dataset):

    def __init__(self, data, node_features):
        self.data = data
        self.node_features = node_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[0]
        return self.transform_graph(datapoint)

    def transform_graph(self, datapoint):
        edge_index = datapoint.edge_index
        edge_type = datapoint.edge_type
        num_relations = len(edge_type.unique())
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


def graph_train(model, trainloader, num_epochs):
    opti = torch.optim.Adam(model.parameters(), 1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for datapoint in tqdm(trainloader):
            x = datapoint.x
            adj = datapoint.adj
            sub_adj = datapoint.sub_adj

            x = x.to(device)
            adj = adj.to(device)
            sub_adj = sub_adj.to(device)
            # y = y.to(device)
            opti.zero_grad()
            y_pred = model(x, sub_adj, adj)

            target_edge_index = datapoint.edge_index
            y = datapoint.edge_type

            target_edge_index = target_edge_index.to(device)
            y = y.to(device)

            y_pred_types = predict_relations_for_edges(target_edge_index, y_pred, datapoint.num_relations, datapoint.num_nodes)

            loss = loss_fn(y_pred_types, y.view(-1))
            loss.backward()
            opti.step()
            total_loss += loss.cpu().item()

            # free memory
            del loss, x, adj, sub_adj, y_pred, target_edge_index, y, y_pred_types

        print(f'{epoch}: {total_loss/len(trainloader)}')
    return model

def graph_eval(model, testloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for datapoint in tqdm(testloader):
            x = datapoint.x
            adj = datapoint.adj
            sub_adj = datapoint.sub_adj

            x = x.to(device)
            adj = adj.to(device)
            sub_adj = sub_adj.to(device)
            y_pred = model(x, sub_adj, adj)

            target_edge_index = datapoint.edge_index
            y = datapoint.edge_type
            target_edge_index = target_edge_index.to(device)
            y = y.to(device)

            y_pred_types = predict_relations_for_edges(target_edge_index, y_pred, datapoint.num_relations, datapoint.num_nodes)

            total += target_edge_index.shape[1]
            correct += (y_pred_types.argmax(dim=1) == y).sum()
        print(f'acc: {correct/total}')


def predict_relations_for_edges(target_edge_index, y_pred, num_relations, num_nodes):
    """From the num_relations many subgraphs, obtain cosine similarity as measure of whether nodes are in relation"""
    x_from = y_pred[:, target_edge_index[0]]
    x_to = y_pred[:, target_edge_index[1]]

    had = x_from * x_to # (n, points, feat_dim)
    had = had.mean(dim=2).T
    return had

def get_dataloader(train, feature_dim=10):
    split = 'train' if train else 'test'
    data = tg.datasets.FB15k_237('./data', split=split)
    return GraphStackFB15K(data, feature_dim)

if __name__ == '__main__':
    data = get_dataloader
    from dsg.DSG import DSGraphG

    model = DSGraphG(10, 10, 10, 2, 1, 10)

    model = graph_train(model, data, 1)
    graph_eval(model)