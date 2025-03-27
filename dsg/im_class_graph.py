import torch
import torch_geometric as tg
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import transforms

from random import choice, randint

class ImageClassData(Dataset):

    def __init__(self, n, data, size, num_classes):
        dataclasses = get_classes(data, num_classes)
        self.data = [generate(data, size, dataclasses) for _ in range(n)]
        self.len = n

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index]

def get_classes(data, num_classes):
    classes = [[] for _ in range(num_classes)]
    for x, cls in data:
        classes[cls].append(x)
    return classes

def generate(data, size, classes):
    x = []
    edges = []
    for i in range(size):
        # start with one element
        if i < 2:
            x.append(choice(data))
        else:
            connect_to = randint(0,i-1)

            # add image with class label
            if randint(0,1) == 0:
                cls = x[connect_to][1]
                x.append((choice(classes[cls]), cls))
            else:
                x.append(choice(data))

            edges.append((connect_to, i))
            edges.append((i, connect_to))
        
            if randint(0,3) == 0:
                connect_to_rnd = randint(0, i-1)
                if connect_to_rnd != connect_to:
                    edges.append((connect_to_rnd, i))
                    edges.append((i, connect_to_rnd))
    
    # to tensors
    y = torch.tensor([cls for im, cls in x])
    x = torch.stack([im for im, cls in x])
    i = torch.tensor(edges)
    edges = torch.sparse_coo_tensor(i.T, torch.ones((i.shape[0],)), (size, size))

    y = _count_neighbors(edges, y)

    return x, edges.to_dense(), y.long()


def _count_neighbors(edges, y):
    indices = edges.coalesce().indices()
    source_nodes = indices[0]
    target_nodes = indices[1]
    
    source_classes = y[source_nodes]
    target_classes = y[target_nodes]
    
    same_class_edges = (source_classes == target_classes)
    
    n_nodes = edges.shape[0]
    same_class_neighbors = torch.zeros(n_nodes, dtype=torch.float, device=edges.device)
    total_neighbors = torch.zeros(n_nodes, dtype=torch.float, device=edges.device)
    
    same_class_counts = same_class_edges.float()
    same_class_neighbors = torch.scatter_add(same_class_neighbors, 0, source_nodes, same_class_counts)
    total_neighbors = torch.scatter_add(total_neighbors, 0, source_nodes, torch.ones_like(same_class_counts))
    
    has_neighbors = total_neighbors > 0
    ratio = torch.zeros_like(same_class_neighbors)
    ratio[has_neighbors] = same_class_neighbors[has_neighbors] / total_neighbors[has_neighbors]
    return ratio >= 0.5

def get_dataloader(train, graph_size, mnist=True, num_graphs=1000):
    if mnist:
        transform_mnist = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        data = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform_mnist)
        data = ImageClassData(n=num_graphs, data=data, size=graph_size, num_classes=10)

        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=train, num_workers=2)
    else:
        transform_cifar = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_cifar)
        data = ImageClassData(n=num_graphs, data=data, size=graph_size, num_classes=10)

        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=train, num_workers=2)

    return loader


if __name__ == '__main__':
    trainloader = get_dataloader(True, 10)
    testloader = get_dataloader(False, 10)

    from dsg.DSG_paper import DSImageG

    model = DSImageG(1, 32, 2, 3, 28, 28, use_agg_fn=False, use_LH2=True, use_LH4=True)

    # from dsg.siamese import SiameseImage

    # model = SiameseImage(1, 32, 2, 3, 28, 28)
    
    from dsg.train import train, eval

    model = train(model, trainloader)
    eval(model, testloader)

