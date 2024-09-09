# 100 Days of Machine Learning

Welcome to my 100 Days of Machine Learning journey! This README will serve as a daily log and resource guide as I progress through various topics and projects in machine learning, with a focus on using PyTorch. Today is **Day 9**.

---

## Day 9: A peek into Distributed and Parallel Training Tutorials



---

## Day 8: More about torch utils data


---

## Day 7: Regularization 
I read about Regularization in machine learning.

---

## Day 6: GPUs
Today read about and studied about GPUs, how they work, their functionality and their role in training machine learning models.

---

## Day 5: Transforms

Data does not always come in its final processed form that is required for training machine learning algorithmms. We use `transforms` to perform some manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -`transform` to modify the features and `target_transform` to modify the labels-that accept callables containing the transformation logic. The `torchvision.transforms` module offers serveral commonly-used transforms out of the box.

The FashionMNIST features are in PIL image format, and the labels are integers. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use `ToTensors` and `Lambda`.

## Build the Neural Network
Neural networks comprise of layers/modules that perform operations on data. The `torch.nn` namespace provides all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the `nn.Module`. A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.

In the following sections, we'll build a neural network to classify images in the FashionMNIST dataset.

---

## Day 4: Datasets and Dataloaders  

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability. Pytorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. `Dataset` stores the samples and their corressponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass `torch.utils.data.Dataset` and implement functions specific to the particular data. They can be used to prototype and benchmark your model. You can be used to prototype and benchmark your model.

---

## Day 3: Deep Dive into Tensors and the `@` Operator

### The Importance of the `@` Operator in PyTorch
- **Matrix Multiplication:** The `@` operator simplifies matrix multiplication for 2D tensors, making your code more readable.
- **Matrix-Vector Multiplication:** Easily multiply a matrix by a vector using the `@` operator.
- **Dot Product:** Compute the dot product between two vectors with a single `@` operation.

This operator is a game-changer for making matrix operations more intuitive compared to `torch.matmul()`.

---

## Day 2: Continuing with Tensors and PyTorch Basics

### Understanding Tensors
Tensors are the backbone of PyTorch. They're similar to NumPy arrays but are optimized for GPU operations and automatic differentiation. Whether you're dealing with data inputs, outputs, or model parameters, tensors will be your go-to data structure.

- **Flexibility:** Tensors can seamlessly interact with NumPy arrays, often sharing memory without data copying.
- **Optimization:** Designed for GPU acceleration and automatic differentiation, making them ideal for deep learning.

---

## Day 1: Getting Started with PyTorch and FashionMNIST

### Quickstart with PyTorch and FashionMNIST
The first step in my journey was a basic introduction to PyTorch using the FashionMNIST dataset. This popular dataset, available via `torchvision`, is a great way to start experimenting with image classification.

- **Link to Torchvision Dataset:** [Click here](https://pytorch.org/vision/stable/datasets.html)

### Loading the Dataset
The `DataLoader` in PyTorch makes it easy to handle large datasets. By passing our dataset as an argument to `DataLoader`, we can efficiently batch, shuffle, and load data using multiple processes.

- **Batch Size:** In my example, I used a batch size of 64, which is a common choice for training deep learning models.

---

## Useful Tips and Commands

### How to Check for a Particular GPU

1. **Using the `lspci` Command:**
   - To list all PCI devices, including the GPU:
     ```bash
     lspci | grep -i vga
     ```
   - For more detailed information:
     ```bash
     lspci -v | grep -i vga
     ```

These commands are essential for confirming the presence and details of a GPU in your system, which is crucial when working with deep learning models that benefit from GPU acceleration.

---

## Goals and Expectations

This README will evolve as I progress through the 100 days, documenting my learning, challenges, and achievements. Whether you’re following along or just browsing, I hope you find this resource helpful!

---
