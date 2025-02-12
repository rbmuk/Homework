import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def avg_large_coordinate(A: torch.Tensor) -> float:
    d = A.shape[1]
    device = A.device
    combinations = torch.cartesian_prod(*([torch.tensor([-1., 1.], device=device)] * d))
    return torch.mean(torch.norm(A @ combinations.t(), dim=0, p=float('inf')))

def sample(n: int, d: int) -> torch.Tensor:
    A = torch.randn(n, d).to(device)
    A /= torch.norm(A, dim=1, keepdim=True)
    return A

def stochastic_gradient_ascent(A: torch.Tensor, learning_rate: float, num_iterations: int) -> torch.Tensor:
    n, d = A.shape
    
    best_A = A.clone().to(device)
    best_avg_largest_coordinate = avg_large_coordinate(A)
    
    for _ in range(num_iterations):
        A += learning_rate * sample(n, d)
        A /= torch.norm(A, dim=1, keepdim=True)
        
        current_avg_largest_coordinate = avg_large_coordinate(A)
        if current_avg_largest_coordinate > best_avg_largest_coordinate:
            best_A = A.clone().to(device)
            best_avg_largest_coordinate = current_avg_largest_coordinate
    
    return best_A


root2 = torch.sqrt(torch.tensor(2.0, device='cuda:0'))
n, d = 2,3
A = torch.tensor([[1/root2, 1/root2, 0], [-1/root2, 1/root2, 0]], device='cuda:0')
print(f"Thought optimal: {avg_large_coordinate(A)}")


# num_tests = 100
# learning_rate = 0.01
# num_iterations = 10000
# A = sample(n,d)
# for _ in range(num_tests):
#     A = stochastic_gradient_ascent(A, learning_rate, num_iterations)
#     print(f"result: {A}, error: {avg_large_coordinate(A)}")

# Best 2x5 case: tensor([[-0.0078, -0.0155, -0.6870,  0.7255,  0.0366],
#                       [-0.0134, -0.0184,  0.7240,  0.6888,  0.0296]], device='cuda:0'), 
# error: 1.4126498699188232 ~ sqrt(2) !!!

# Best 2x3 case: tensor([[-0.7041,  0.7100, -0.0068],
#                       [-0.7247, -0.6890,  0.0119]], device='cuda:0'), 
# error: 1.4139163494110107 ~ sqrt(2) !!!

# Maybe for the 2xn case the error always <= sqrt(2) ???

# Current best 3x5 case:
#tensor([[-0.0031,  0.0077, -0.5795, -0.5696,  0.5829],
#        [-0.0076, -0.0025,  0.5888,  0.5682,  0.5748],
#        [-0.0013, -0.0012,  0.7081, -0.7061, -0.0018]], device='cuda:0')
# with error 1.5730317831039429 ~ pi/2 (1.5707963267948966) Seems to be a local max