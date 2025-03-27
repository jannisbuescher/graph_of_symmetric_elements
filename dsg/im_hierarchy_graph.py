import torch
import torchvision.transforms as transforms
import torch_geometric as tg
from torch.utils.data import Dataset

import torchvision

import random

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class HierarchicalImageFeatures(Dataset):

    def __init__(self, data, depth):
        self.dataset = data
        self.depth = depth

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        x, adj = img_hierarchy_transform(img, self.depth)
        return x, adj, target
    

def img_hierarchy_transform(image, depth):
    """Image is partitioned into equal parts. The whole image is connected to the parts in the graph.
    This is done recursively to depth. Nodes are randomly reordered."""

    all_parts = []
    edges = []
    size_h, size_w = image.shape[-2:]
    size_h_part, size_w_part = size_h // (2 ** depth), size_w // (2 ** depth)
    resize = transforms.Resize((size_h_part, size_w_part))
    
    def partition_image(img, current_depth, parent_idx=None):
        current_idx = len(all_parts)
        all_parts.append(resize(img))
        
        if parent_idx is not None:
            edges.append((parent_idx, current_idx))
        
        if current_depth >= depth:
            return [current_idx]
        
        _, h, w = img.shape
        h_mid = h // 2
        w_mid = w // 2
        
        quadrants = [
            img[:, :h_mid, :w_mid],
            img[:, :h_mid, w_mid:],
            img[:, h_mid:, :w_mid],
            img[:, h_mid:, w_mid:]
        ]
        
        child_indices = []
        for quadrant in quadrants:
            indices = partition_image(quadrant, current_depth + 1, current_idx)
            child_indices.extend(indices)
    
        random.shuffle(child_indices)
        return [current_idx] + child_indices
    
    partition_image(image, 0)
    
    x = torch.stack(all_parts)
    
    n = len(all_parts)
    adj = torch.zeros((n, n), dtype=torch.float)
    for i, j in edges:
        adj[i, j] = 1.
        adj[j, i] = 1.
        
    return x, adj

def get_dataloader(train, depth):
    if train:
        data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    else:
        data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    data = HierarchicalImageFeatures(data, depth)
    return torch.utils.data.DataLoader(data, batch_size=64,
                                        shuffle=train, num_workers=2)


if __name__ == "__main__":
    trainloader = get_dataloader(train=True, depth=2)
    for x, adj, y in trainloader:
        print(x.shape, adj.shape, y.shape)
        break