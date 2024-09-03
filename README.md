# 100_days-ML

## How to check for a particular GPU

## 1. Using 'lspci' Command
The 'lspci' command lists all PCI devices in your system, including the GPU

> Open a terminal and run:
```bash
lspci | grep -i vga
```
This will display information about the GPU, if present. The output should show something like 'VGA compatible controller' followed by the GPU model.

> For more detailed information:
```bash
lspci -v | grep -i vga
```

## Quickstart
This is a basic intro to pytorch and the fashionmnist dataset

*Link to torchvision dataset*
[click here](https://pytorch.org/vision/stable/datasets.html)

## About moving the dataset
We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.

# Tensors
Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data. Tensors are close optimized for automatic differentiation (we'll see more about that later in the Autograd section). If you are familiar with ndarrays, you'll be right at home with the Tensor API.

# Day 3
## Day 2 of a quick intro to pytorch and a continuation of tensors  

### The importance of an `@` operator:  
* `@` performs matrix multiplication when used with 2D tensors (matrices).
* It performs matrix-vector multiplication when used with a matrix and a vector.
* It computes the dot product when used with two vectors.

This operator simplifies and clarifies the syntax for matrix operations in PyTorch, making code more readable and concise compared to using functions like torch.matmul().
