if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    history: Dict[str, Any] = {}
    models = {'linear': nn.Sequential(
                  LinearLayer(2, 2, RNG), 
                  SoftmaxLayer()),
              'sigmoid': nn.Sequential(
                  LinearLayer(2, 2, RNG),
                  SigmoidLayer(),
                  LinearLayer(2, 2, RNG),
                  SoftmaxLayer()),
              'relu': nn.Sequential(
                  LinearLayer(2, 2, RNG),
                  ReLULayer(),
                  LinearLayer(2, 2, RNG),
                  SoftmaxLayer()),
              'sigmoid relu': nn.Sequential(
                  LinearLayer(2, 2, RNG), 
                  SigmoidLayer(), 
                  LinearLayer(2, 2, RNG),
                  ReLULayer(), 
                  LinearLayer(2, 2, RNG),
                  SoftmaxLayer()),
              'relu sigmoid': nn.Sequential(
                  LinearLayer(2, 2, RNG), 
                  ReLULayer(), 
                  LinearLayer(2, 2, RNG),
                  SigmoidLayer(), 
                  LinearLayer(2, 2, RNG),
                  SoftmaxLayer())}
    for name, model in models.items():
        history[name] = train_model(model, dataset_train, dataset_val)
        history[name]['model'] = model
    return history

def train_model(model: nn.Module, dataset_train, dataset_val) -> dict:
    criterion = CrossEntropyLossLayer()
    optimizer = SGDOptimizer(model.parameters(), lr=1e-1)
    hist = train(dataset_train, model, criterion, optimizer, dataset_val, epochs=250)
    return hist


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    running_loss = 0
    for data in iter(dataloader):
        inputs, labels = data
        outputs = model(inputs)
        running_loss += torch.mean((labels == torch.argmax(outputs, dim=1)).to(dtype=torch.float))
    return running_loss.item() / len(dataloader)


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=True)

    models = {'linear', 
                'sigmoid',
                'relu',
                'sigmoid relu',
                'relu sigmoid'}
    #save_models(models, dataloader_train, dataloader_val)

    model = torch.load('data/models/ce/relu sigmoid')
    print(accuracy_score(model, dataloader_test))
    plot_model_guesses(dataloader_test, model, 'Test accuracy')

def save_models(models, dataset_train, dataset_val):
    
    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)
    for model in models:
        plt.plot(ce_configs[model]['train'], label=f'{model} train')
        plt.plot(ce_configs[model]['val'], label=f'{model} val')
    plt.title('CrossEntropy Loss for different models')
    plt.legend()
    plt.show()

    for model in models:
        torch.save(ce_configs[model]['model'], f'data/models/ce/{model}')

if __name__ == "__main__":
    main()
