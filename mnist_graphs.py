import torch
import torchvision
import torchvision.transforms as transforms
import torch_geometric as tg
from torch.utils.data import Dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64


def get_dataloader(train):
    if train:
        data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    else:
        data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    data = AugmentedCIFAR10(data)
    return torch.utils.data.DataLoader(data, batch_size=batch_size,
                                        shuffle=train, num_workers=2)

class AugmentedCIFAR10(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img, adj, target = img_transform_graph(img)
        return img, adj, target


def img_transform_graph(image):
    """Take image dataset and transform into a graph of transformations. Nodes are randomly ordered."""
    im0 = image
    im90d = transforms.functional.rotate(image, 90)
    im180d = transforms.functional.rotate(image, 180)
    im270d = transforms.functional.rotate(image, 270)
    im_flip = transforms.functional.hflip(im0)
    im90f = transforms.functional.rotate(im_flip, 90)
    im180f = transforms.functional.rotate(im_flip, 180)
    im270f = transforms.functional.rotate(im_flip, 270)
    x = torch.stack([im0, im90d, im180d, im270d, im_flip, im90f, im180f, im270f])
    target = torch.tensor([0,1,2,3,4,5,6,7])
    # adj = torch.tensor([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7],
    #                               [1,3,4,0,2,5,1,3,6,2,4,7,3,5,0,4,6,1,5,7,2,6,0,3]])
    # adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]))
    adj = torch.tensor([[0,1,0,1,1,0,0,0],
                        [1,0,1,0,0,1,0,0],
                        [0,1,0,1,0,0,1,0],
                        [1,0,1,0,0,0,0,1],
                        [1,0,0,0,0,1,0,1],
                        [0,1,0,0,1,0,1,0],
                        [0,0,1,0,0,1,0,1],
                        [0,0,0,1,1,0,1,0],
                        ])
    perm = torch.randperm(8)
    x = x[perm]
    target = target[perm]
    adj = adj[perm][:,perm]
    return x, adj, target