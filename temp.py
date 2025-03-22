import torch

# b n d h w
x = torch.rand((64, 4, 8, 2, 2))
adj = torch.zeros((64, 4, 4))
adj[0,0,1] = 1
adj[0,0,2] = 1
adj[0,0,3] = 1
adj[0,2,3] = 1
adj[0,1,0] = 1
adj[0,2,0] = 1
adj[0,3,0] = 1
adj[0,3,2] = 1

y = torch.einsum('bnm,bmdhw->bndhw', adj, x)
print(y)