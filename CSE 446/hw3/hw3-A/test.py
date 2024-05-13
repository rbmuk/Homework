import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([0, 0, 1])
print(x)
print(torch.argmax(x) == torch.argmax(y))