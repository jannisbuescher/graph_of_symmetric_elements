import torch
from tqdm import tqdm

from dsg.im_transform_graphs import get_dataloader
from dsg.im_hierarchy_graph import get_dataloader as get_hier_dataloader
from dsg.DSG import DSImageG

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, trainloader, num_epochs=10):
    opti = torch.optim.Adam(model.parameters(), 1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for x, adj, y in tqdm(trainloader):
            x = x.to(device)
            adj = adj.to(device)
            y = y.to(device)
            opti.zero_grad()
            y_pred = model(x, adj)
            loss = loss_fn(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
            loss.backward()
            opti.step()
            total_loss += loss.cpu().item()
        print(f'{epoch}: {total_loss/len(trainloader)}')
    return model

def eval(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, adj, y in tqdm(testloader):
            x = x.to(device)
            adj = adj.to(device)
            y = y.to(device)
            y_pred = model(x, adj)
            if len(y_pred.shape) == 2:
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
            elif len(y_pred.shape) == 3:
                _, predicted = torch.max(y_pred.data, 2)
                total += y.size(0) * y.size(1)
            else:
                raise Exception()
            correct += (predicted == y).sum().item()
    print(f'Accuracy: {correct/total}')
    return correct/total

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

            target_edge_index = datapoint.target_edge_index
            y = datapoint.target_edge_type
            y_pred_types = predict_relations_for_edges(target_edge_index, y_pred, datapoint.num_relations, datapoint.num_nodes)

            loss = loss_fn(y_pred_types, y.view(-1))
            loss.backward()
            opti.step()
            total_loss += loss.cpu().item()
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

            target_edge_index = datapoint.target_edge_index
            y = datapoint.target_edge_type
            y_pred_types = predict_relations_for_edges(target_edge_index, y_pred, datapoint.num_relations, datapoint.num_nodes)

            total += target_edge_index.shape[1]
            correct += (y_pred_types.argmax(dim=1) == y).sum()
        print(f'acc: {correct/total}')
    return model

def predict_relations_for_edges(target_edge_index, y_pred, num_relations, num_nodes):
    """From the num_relations many subgraphs, obtain cosine similarity as measure of whether nodes are in relation"""
    x_from = y_pred[:, target_edge_index[0]]
    x_to = y_pred[:, target_edge_index[1]]
    top = torch.einsum('nmd,nmd->nm', x_from, x_to)
    norm = torch.mul(torch.linalg.norm(x_from, dim=2), torch.linalg.norm(x_to, dim=2))
    res = torch.div(top, norm)
    return res.view(-1, num_relations)

if __name__ == '__main__':
    
    # model = DSImageG(1, 8, 8, 3, 1, 28, 28).to(device)
    # trainloader = get_dataloader(train=True, dataset='MNIST')

    # model = train(model, trainloader, 10)
    # eval(model, get_dataloader(train=False, dataset='MNIST'))

    model = DSImageG(3, 8, 10, 3, 1, 8, 8, False)
    trainloader = get_hier_dataloader(True, 2)

    model = train(model, trainloader, 1)
    eval(model, get_hier_dataloader(False, 2))