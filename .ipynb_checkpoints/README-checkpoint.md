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