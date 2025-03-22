import torch
from tqdm import tqdm

from dsg.mnist_graphs import get_dataloader
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
            loss = loss_fn(y_pred, y)
            loss.backward()
            opti.step()
            total_loss += loss.cpu().item()
        print(f'{epoch}: {total_loss}')
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
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy: {correct/total}')
    return correct/total


if __name__ == '__main__':
    
    model = DSImageG(3, 8, 8, 3, 1, 32, 32).to(device)
    trainloader = get_dataloader(train=True)

    model = train(model, trainloader, 1)
            