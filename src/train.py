import torch

from mnist_graphs import get_dataloader
from DSG import DSImageG


def train(model, trainloader, num_epochs=10):
    opti = torch.optim.Adam(model.parameters(), 1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for x, adj, y in trainloader:
            opti.zero_grad()
            y_pred = model(x, adj)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opti.step()
            total_loss += loss.item()
            print(f'{loss.item()}')
        print(f'{epoch}: {total_loss}')
    return model


if __name__ == '__main__':
    
    model = DSImageG(3, 8, 8, 3, 1, 32, 32)
    trainloader = get_dataloader(train=True)

    model = train(model, trainloader, 1)
            