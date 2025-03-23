import torch
import torch_geometric as tg

from torch.utils.data import Dataset

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ToyGraphs(Dataset):

    def __init__(self):
        self.len = 10

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        adj = torch.randint(0,2,(10,10))
        adj = torch.matmul(adj.T, adj)
        adj = torch.clamp(adj, max=1)
        adj.fill_diagonal_(0)
        if index < self.len // 2:
            big_adj = torch.block_diag(*([adj] * 3))
            x = torch.ones((3,10,1))
            y = 0
            
            # meta_adj = torch.randint(0,2,(3,3))
            # meta_adj = torch.matmul(meta_adj.T, meta_adj)
            # meta_adj = torch.clamp(meta_adj, max=1)
            # meta_adj.fill_diagonal_(0)

            meta_adj = torch.ones((3,3))
            meta_adj.fill_diagonal_(0)

        else:
            other_adj = torch.randint(0,2,(10,10))
            other_adj = torch.matmul(other_adj.T, other_adj)
            other_adj = torch.clamp(other_adj, max=1)
            other_adj = other_adj - torch.eye(other_adj.shape[0])

            big_adj = torch.block_diag(*([adj] * 2 + [other_adj]*1))
            x = torch.ones((3,10,1))
            y = 1

            # meta_adj = other_adj

            meta_adj = torch.zeros((3,3))
            meta_adj[0,1] = 1
            meta_adj[1,0] = 1
        
        # meta_adj = torch.randint(0,2,(3,3))
        # meta_adj = torch.matmul(meta_adj.T, meta_adj)
        # meta_adj = torch.clamp(meta_adj, max=1)
        # meta_adj.fill_diagonal_(0)

        return tg.data.Data(x=x, sub_adj=tg.utils.dense_to_sparse(big_adj)[0], adj=meta_adj, y=y)
    
def get_dataloader():
    data = ToyGraphs()
    return tg.loader.DataLoader(data)

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
            y = datapoint.y

            x = x.to(device)
            adj = adj.to(device)
            sub_adj = sub_adj.to(device)
            y = y.to(device)
            opti.zero_grad()
            y_pred = model(x, sub_adj, adj)

            loss = loss_fn(y_pred.sum(dim=0).view(1,-1), y)
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
            y = datapoint.y

            x = x.to(device)
            adj = adj.to(device)
            sub_adj = sub_adj.to(device)
            y = y.to(device)
            y_pred = model(x, sub_adj, adj)

            total += y.shape[0]
            correct += (y_pred.sum(dim=0).argmax() == y).sum()
        print(f'acc: {correct/total}')
    return model

if __name__ == '__main__':
    from dsg.DSG import DSGraphG

    model = DSGraphG(1, 128, 2, 3, 1, 3, False)

    data = get_dataloader()

    model = graph_train(model, data, 100)

    graph_eval(model, data)