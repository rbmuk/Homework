import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.argmax(x, dim=1).view(-1,1)
print(torch.gather(x, 1, y))