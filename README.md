# 100 Days of Machine Learning

Welcome to my 100 Days of Machine Learning journey! This README will serve as a daily log and resource guide as I progress through various topics and projects in machine learning, with a focus on using PyTorch. Today is **Day 50**

---

## Day 50: Tensorboard and Wandb

---

## Day 49: Compared a ResNet vs ConvNeXt
[Reference Repo](https://keras.io/api/keras_cv/models/)

---

## Day 48: Looked into ConvNeXt, the ConvNet of 2020s
[!Reference Repo](https://github.com/facebookresearch/ConvNeXt)

---

## Day 47: Final touches with Autograd

---

## Day 46: Autograd mechanics
This note will present an overview of how autograd works and records the operations. It's not strictly necessary to understand all this but I am recommended to get familiar with it, as it will help me write more efficient, cleaner programs, and can aid me in debugging.

## How autograd encodes the history
Autograd is a reverse automatic differentiation system. Conceptually, autograd records a graph recording all of the operations that created the data as you execute operations, giving you a directed acyclic graph whose leaves are the input tensors and roots are the ouput tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

Internally, autograd represents this graph as a graph of `Function` objects (really expressions), which can be `apply()` ed to compute the result of evaluating the graph. When computing the forward pass, autograd simultaneously performs the requested computations and builds up a graph representing the function that computes the gradients (the `.grad_fn` attribute of each `torch.Tensor` is an entry point into this graph). When the forward pass is completed, we evaluate this graph in the backwards pass to compute the gradients.

An important thing to note is that the graph is recreated from scratch at every iteration, and this is exactly what allows for using arbitrary Python control flow statements, that can change the overall shape and size of the graph at every iteration. You don't have to encode all possible paths before you launch the training - what you run is what you differentiate.

## Saved tensors
Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass. For example, the function `x -> x^2` saves the input x to compute the gradient.

When defining a custom Python `Function`, you can use `save_for_backward()` to save tensors during the forward pass and `saved_tensors` to retrieve them during the backward pass.

For operations that PyTorch defines (e.g. `torch.pow()`), tensors are automatically saved as needed. You can explore (for educational or debugging purposes) which tensors are saved by a certain `grad_fn` by looking for its attributes starting with the prefix `_saved`.

```python
x = torch.randn(5, requires_grad=True)
y = x.exp()
print(y.equal(y.grad_fn._saved_result))  # True
print(y is y.grad_fn._saved_result)  # False
```

Under the hood, to prevent reference cycles, PyTorch has packed the tensor upon saving and unpacked it into a different tensor for reading. Here, the tensor you get from accessing y.grad_fn._saved_result is a different tensor object than y (but they still share the same storage).

Whether a tensor will be packed into a different tensor object depends on whether it is an output of its own grad_fn, which is an implementation detail subject to change and that users should not rely on.

You can control how PyTorch does packing / unpacking with Hooks for saved tensors.
Gradients for non-differentiable functions

The gradient computation using Automatic Differentiation is only valid when each elementary function being used is differentiable. Unfortunately many of the functions we use in practice do not have this property (relu or sqrt at 0, for example). To try and reduce the impact of functions that are non-differentiable, we define the gradients of the elementary operations by applying the following rules in order:

- If the function is differentiable and thus a gradient exists at the current point, use it.

- If the function is convex (at least locally), use the sub-gradient of minimum norm (it is the steepest descent direction).

- If the function is concave (at least locally), use the super-gradient of minimum norm (consider -f(x) and apply the previous point).

- If the function is defined, define the gradient at the current point by continuity (note that inf is possible here, for example for sqrt(0)). If multiple values are possible, pick one arbitrarily.

- If the function is not defined (sqrt(-1), log(-1) or most functions when the input is NaN, for example) then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). Most functions will use NaN as the gradient, but for performance reasons, some functions will use other values (log(-1), for example).

- If the function is not a deterministic mapping (i.e. it is not a mathematical function), it will be marked as non-differentiable. This will make it error out in the backward if used on tensors that require grad outside of a no_grad environment.

Locally disabling gradient computation

There are several mechanisms available from Python to locally disable gradient computation:

To disable gradients across entire blocks of code, there are context managers like no-grad mode and inference mode. For more fine-grained exclusion of subgraphs from gradient computation, there is setting the requires_grad field of a tensor.

Below, in addition to discussing the mechanisms above, we also describe evaluation mode (nn.Module.eval()), a method that is not used to disable gradient computation but, because of its name, is often mixed up with the three.
Setting requires_grad

requires_grad is a flag, defaulting to false unless wrapped in a nn.Parameter, that allows for fine-grained exclusion of subgraphs from gradient computation. It takes effect in both the forward and backward passes:

During the forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad. During the backward pass (.backward()), only leaf tensors with requires_grad=True will have gradients accumulated into their .grad fields.

It is important to note that even though every tensor has this flag, setting it only makes sense for leaf tensors (tensors that do not have a grad_fn, e.g., a nn.Module’s parameters). Non-leaf tensors (tensors that do have grad_fn) are tensors that have a backward graph associated with them. Thus their gradients will be needed as an intermediary result to compute the gradient for a leaf tensor that requires grad. From this definition, it is clear that all non-leaf tensors will automatically have require_grad=True.

Setting requires_grad should be the main way you control which parts of the model are part of the gradient computation, for example, if you need to freeze parts of your pretrained model during model fine-tuning.

To freeze parts of your model, simply apply .requires_grad_(False) to the parameters that you don’t want updated. And as described above, since computations that use these parameters as inputs would not be recorded in the forward pass, they won’t have their .grad fields updated in the backward pass because they won’t be part of the backward graph in the first place, as desired.

Because this is such a common pattern, requires_grad can also be set at the module level with nn.Module.requires_grad_(). When applied to a module, .requires_grad_() takes effect on all of the module’s parameters (which have requires_grad=True by default).
Grad Modes

Apart from setting requires_grad there are also three grad modes that can be selected from Python that can affect how computations in PyTorch are processed by autograd internally: default mode (grad mode), no-grad mode, and inference mode, all of which can be togglable via context managers and decorators.

Mode
	

Excludes operations from being recorded in backward graph
	

Skips additional autograd tracking overhead
	

Tensors created while the mode is enabled can be used in grad-mode later
	

Examples

default
			

✓
	

Forward pass

no-grad
	

✓
		

✓
	

Optimizer updates

inference
	

✓
	

✓
		

Data processing, model evaluation
Default Mode (Grad Mode)

The “default mode” is the mode we are implicitly in when no other modes like no-grad and inference mode are enabled. To be contrasted with “no-grad mode” the default mode is also sometimes called “grad mode”.

The most important thing to know about the default mode is that it is the only mode in which requires_grad takes effect. requires_grad is always overridden to be False in both the two other modes.
No-grad Mode

Computations in no-grad mode behave as if none of the inputs require grad. In other words, computations in no-grad mode are never recorded in the backward graph even if there are inputs that have require_grad=True.

Enable no-grad mode when you need to perform operations that should not be recorded by autograd, but you’d still like to use the outputs of these computations in grad mode later. This context manager makes it convenient to disable gradients for a block of code or function without having to temporarily set tensors to have requires_grad=False, and then back to True.

For example, no-grad mode might be useful when writing an optimizer: when performing the training update you’d like to update parameters in-place without the update being recorded by autograd. You also intend to use the updated parameters for computations in grad mode in the next forward pass.

The implementations in torch.nn.init also rely on no-grad mode when initializing the parameters as to avoid autograd tracking when updating the initialized parameters in-place.
Inference Mode

Inference mode is the extreme version of no-grad mode. Just like in no-grad mode, computations in inference mode are not recorded in the backward graph, but enabling inference mode will allow PyTorch to speed up your model even more. This better runtime comes with a drawback: tensors created in inference mode will not be able to be used in computations to be recorded by autograd after exiting inference mode.

Enable inference mode when you are performing computations that don’t need to be recorded in the backward graph, AND you don’t plan on using the tensors created in inference mode in any computation that is to be recorded by autograd later.

It is recommended that you try out inference mode in the parts of your code that do not require autograd tracking (e.g., data processing and model evaluation). If it works out of the box for your use case it’s a free performance win. If you run into errors after enabling inference mode, check that you are not using tensors created in inference mode in computations that are recorded by autograd after exiting inference mode. If you cannot avoid such use in your case, you can always switch back to no-grad mode.

For details on inference mode please see Inference Mode.

For implementation details of inference mode see RFC-0011-InferenceMode.
Evaluation Mode (nn.Module.eval())

Evaluation mode is not a mechanism to locally disable gradient computation. It is included here anyway because it is sometimes confused to be such a mechanism.

Functionally, module.eval() (or equivalently module.train(False)) are completely orthogonal to no-grad mode and inference mode. How model.eval() affects your model depends entirely on the specific modules used in your model and whether they define any training-mode specific behavior.

You are responsible for calling model.eval() and model.train() if your model relies on modules such as torch.nn.Dropout and torch.nn.BatchNorm2d that may behave differently depending on training mode, for example, to avoid updating your BatchNorm running statistics on validation data.

It is recommended that you always use model.train() when training and model.eval() when evaluating your model (validation/testing) even if you aren’t sure your model has training-mode specific behavior, because a module you are using might be updated to behave differently in training and eval modes.
In-place operations with autograd

Supporting in-place operations in autograd is a hard matter, and we discourage their use in most cases. Autograd’s aggressive buffer freeing and reuse makes it very efficient and there are very few occasions when in-place operations lower memory usage by any significant amount. Unless you’re operating under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:

    In-place operations can potentially overwrite values required to compute gradients.

    Every in-place operation requires the implementation to rewrite the computational graph. Out-of-place versions simply allocate new objects and keep references to the old graph, while in-place operations, require changing the creator of all inputs to the Function representing this operation. This can be tricky, especially if there are many Tensors that reference the same storage (e.g. created by indexing or transposing), and in-place functions will raise an error if the storage of modified inputs is referenced by any other Tensor.

In-place correctness checks

Every tensor keeps a version counter, that is incremented every time it is marked dirty in any operation. When a Function saves any tensors for backward, a version counter of their containing Tensor is saved as well. Once you access self.saved_tensors it is checked, and if it is greater than the saved value an error is raised. This ensures that if you’re using in-place functions and not seeing any errors, you can be sure that the computed gradients are correct.

[!For Tomorrow](https://pytorch.org/docs/stable/notes/autograd.html)

---

## Day 45: Deep dive into PyTorch's Automatic Differentiation

---

## Day 44: Deep dive into C plus plus classes

---

## Day 43: Autograd review

---

## Day 42:

---

## Day 41:

---

## Day 40:

---

## Day 39: Finalized with intermediate docker concepts

---

## Day 38: Deep Dive into Docker

---

## Day 37: Docker and Kubernetes

---

## Day 36: Continuation with ResNet

```bash
(lmri) leo@mcbills:~/Desktop/100_days/100_days-ML/models$ python3 resnet.py 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,472
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,928
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,928
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
    ResidualBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,928
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,928
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
    ResidualBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,856
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,584
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,320
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
    ResidualBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,584
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,584
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
    ResidualBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         295,168
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         590,080
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          33,024
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
    ResidualBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         590,080
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         590,080
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
    ResidualBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,180,160
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,584
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 7, 7]               0
    ResidualBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
    ResidualBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                   [-1, 10]           5,130
================================================================
Total params: 11,186,442
Trainable params: 11,186,442
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 62.79
Params size (MB): 42.67
Estimated Total Size (MB): 106.03
----------------------------------------------------------------
```
---

## Day 35: A deep dive into the ResNet architecture

---

## Day 34:

---

## Day 33:

---

## Day32: Write a script for custom datasets for an image enhancing project

---

## Day 31: Practice with organizing own datasets

---

## Day 30:

---

## Day 29:

---

## Day 28:
---

## Day 27: Logging and progress bars with python

---

## Day 26: Storage Classes and Namespaces
This chapter begins by describing storage classes for objects and functions. The storage class is responsible for defining those parts of a program where an object or function can be used. Namespaces can be used to avoid conflicts when naming global identifiers.

---

## Day 25: Macros in c++

## Converting Arithmetic Types
This chapter introduces implicit type conversions, which are performed in c++ whenever different arithmetic types occur in expressions.
Additionally, an operator for explicit type conversion is introduced.

## The standard Class string
This chapter introduces the standard class string, which is used to represent strings. Besides defining strings we will also look at various methods of string manipulation. These include inserting and erasing, searching and replacing, comparing and concatenating strings.

## Functions
This chapter describes how to write functions of your own. Besides the basic rules, the following topics are discussed:
- passing arguments
- definition of iniline functions
- overloading functions and default arguments
- the principle of recursion

---

## Day 24: Deep dive into control statements


---

## Day 23: Operators for Fundamental Types
Today, I shall look at operators needed for calculations and selections are introduced. Overloading and other operators, such as those needed for bit manipulations, are gonna be tackled later.

## Control Flow
In this chapter, I was introduced to the statements needed to control the flow of a program. These are;
* loops with while, do-while, and for
* selections with if-else, switch, and the conditional operator
* jumps with goto, continue, and break

---

## Day 22: Functions and classes
I looked at functions and classes in c++ basically but did not look at user defined classes

---

## Day 21: Introduction to C++ programming

### How to run c++ in my terminal
1. Write the program and save it as an `.cpp` program
2. Compile the program using the g++ compiler by;
```bash
g++ -o execfile program.cpp
```
3. Run the executable file
```bash
./execfile
```

### Use a Build System (Optional)
If you are working on larger projects, you might want to use `make` or `cmake` to handle the build process. Install them with:
```bash
sudo apt install make cmake
```
You can then set up `Makefiles` or `CMakeLists.txt` for more complex projects.

### Debugging Tools (Optional)
For debugging, you can install `gdb`, the GNU debugger:
```bash
sudo apt install gdb
```

---

Day 20: Read about CUDA Programming 

---

## Day 19: Continuation with the training of the classifier

---

## Day 18: Continuation with the training of the classifier

---

## Day 17: Training a Classifier

---

## Day 16: Read through supervised learning from Andrew Ng notes

---

## Day 15: Last day on an intro to Distributed and Data Parallel programming

---

## Day 14: I researched and wrote an article on how to find and use GPUs in Pytorch

	 	 	 	  
Overcoming GPU Limitations in AI Research: Tips for Resource-Constrained Developers

Hello!

The race for compute power is a hot topic among AI Researchers. The race for compute power is a critical challenge in the world of AI and machine learning. NVIDIA, a $3 trillion company based in Santa Clara, California, is leading this revolution.

For many researchers—especially those in low-resource regions—access to GPUs is a major barrier. But don’t let that stop you!

In this post, I’ll cover the role of GPUs in machine learning, using them with PyTorch, leveraging multiple GPUs, and—most importantly—how to access GPUs without purchasing one, especially if you're from a low-resource region like me.

**So, why are GPUs essential in AI?**  
 GPUs handle the heavy math behind AI, processing vast datasets efficiently thanks to their parallelized cores. While your typical home computer has a CPU with a few cores, GPUs have thousands—making them perfect for training complex machine learning models.

**Using GPUs in PyTorch:**  
 You can easily check for available GPUs in PyTorch with this code:


```python
import torch

device = ("cuda"
          if torch.cuda.is\_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu"
)
print(f"Using {device} device")**

```

Checkout the simple explanation from the [PyTorch documentation](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) on how to move your tensors to the GPU.

Want to scale up? You can leverage [**Distributed and Parallel Training**](https://pytorch.org/tutorials/distributed/home.html) using techniques like DistributedDataParallel (DDP) and Tensor Parallel (TP) to train large models on multiple GPUs.

**No GPU? No problem!**  
 Here are a few free alternatives to get started:

* **Kaggle Kernels**: 	Free access to Nvidia K80 GPUs. [Learn 	more](https://www.kaggle.com/code/dansbecker/running-kaggle-kernels-with-a-gpu).  
   	  
* **Google Colab**: Free GPUs for small projects, 	with Pro options for more power. [Read 	more](https://www.geeksforgeeks.org/how-to-use-gpu-in-google-colab/).

For larger projects, check [**Google Cloud**](https://cloud.google.com/compute/gpus-pricing?_gl=1*152ua1h*_up*MQ..&gclid=CjwKCAjw6JS3BhBAEiwAO9waF1gethq5sbpnf6Nb14gL8alrn4Tr4wr8F6ZAaKuXSiZAmPGRVfcoEhoCJlsQAvD_BwE&gclsrc=aw.ds) pricing or explore platforms like [**vast.ai**](https://vast.ai/) for affordable GPU rentals. Also, check out this blog on the [**Top 10 cloud GPU platforms for deep learning**](https://blog.paperspace.com/top-ten-cloud-gpu-platforms-for-deep-learning/) by Samuel Ozechi.

I hope this helps someone out there facing similar challenges. Keep pushing forward in your AI research, and good luck on your journey\! 💪🚀


---

## Day 13: A continuation with Distributed and Parallel Training Tutorials

> *Note* : *Fun Joke!* *What is the dictionary definition of shard?*
> - (online gaming) An instance of an MMORPG that is one of several independent and structurally identical virtual worlds, none of which has so many players as to exhaust a system's resources.
> - The other is, (database) A component of a sharded distributed database.
> - Synonyms: partition 

---

## Day 12: A continuation with Distributed and Parallel Training Tutorials

---

## Day 11: A continuation with Distributed and Parallel Training Tutorials


---

## Day 10: A continuation with Distributed and Parallel Training Tutorials


---

## Day 9: A peek into Distributed and Parallel Training Tutorials
Distributed training is a model training paradigm that involves spreading training workload across multiple worker nodes, therefore significantly improving the speed of training and model accuracy. While distributed training can be used for any type of ML model training, it is most beneficial to use it for large models and compute demanding tasks as deep learning.


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
