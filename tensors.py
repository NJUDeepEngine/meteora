import torch
import torch.nn.functional as F
x = torch.randn(3, 5)
z = torch.nn.Linear(25,5)
print(z)

t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

indices = torch.tensor([0, 2])

y = torch.index_select(x, 1, indices)
print(z.weight.size())
z = F.linear(x, z.weight[:,:5], None)


print(x)
print(indices)
print(z)
