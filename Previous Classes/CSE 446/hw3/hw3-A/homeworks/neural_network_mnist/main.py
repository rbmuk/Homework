# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        gen_0 = Uniform(-1/math.sqrt(d), 1/math.sqrt(d))
        self.W0 = Parameter(gen_0.sample((h, d)))
        self.b0 = Parameter(gen_0.sample((h, 1)))
        gen_1 = Uniform(-1/math.sqrt(h), 1/math.sqrt(h))
        self.W1 = Parameter(gen_1.sample((k, h)))
        self.b1 = Parameter(gen_1.sample((k, 1)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(x @ self.W0.T + self.b0.T) @ self.W1.T + self.b1.T


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        gen_0 = Uniform(-1/math.sqrt(d), 1/math.sqrt(d))
        self.W0 = Parameter(gen_0.sample((h0, d)))
        self.b0 = Parameter(gen_0.sample((h0, 1)))
        gen_1 = Uniform(-1/math.sqrt(h0), 1/math.sqrt(h0))
        self.W1 = Parameter(gen_1.sample((h1, h0)))
        self.b1 = Parameter(gen_1.sample((h1, 1)))
        gen_2 = Uniform(-1/math.sqrt(h1), 1/math.sqrt(h1))
        self.W2 = Parameter(gen_2.sample((k, h1)))
        self.b2 = Parameter(gen_2.sample((k, 1)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(relu(x @ self.W0.T + self.b0.T) @ self.W1.T + self.b1.T) @ self.W2.T + self.b2.T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(model: Module, train_loader: DataLoader):
    acc = 0
    with torch.no_grad():
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            acc += torch.mean((labels == torch.argmax(model(inputs), dim=1)).to(dtype=float))
    return acc.item() / len(train_loader)

@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    history = []
    model.train()
    while (accuracy(model, train_loader) < 0.99):
        print(accuracy(model, train_loader))
        running_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        history.append(running_loss / len(train_loader))
    return history

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    train_dataloader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=128, shuffle=True)
    model = F1(64, 28 ** 2, 10)
    #model = F2(32, 32, 28 ** 2, 10)
    model = model.to(device)
    total_params = 0
    for parameter in model.parameters():
        curr = 1
        for axis in iter(parameter.shape):
            curr *= axis
        total_params += curr
    print(total_params)
    history = train(model, Adam(model.parameters(), lr=1e-3), train_dataloader)

    plt.plot(history)
    plt.xlabel('epoch')
    plt.ylabel('Training loss')
    plt.show()

    print(f'accuracy on test data: {accuracy(model, test_dataloader)}')
    print(f'loss: {cross_entropy(model(x_test.to(device)), y_test.to(device))}')

if __name__ == "__main__":
    main()
