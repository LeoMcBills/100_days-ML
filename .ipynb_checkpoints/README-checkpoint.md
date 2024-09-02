# 100_days-ML

## How to check for a particular GPU

## 1. Using 'lspci' Command
The 'lspci' command lists all PCI devices in your system, including the GPU

-> Open a terminal and run:
```bash
lspci | grep -i vga
```
This will display information about the GPU, if present. The output should show something like 'VGA compatible controller' followed by the GPU model.

> For more detailed information:
```bash
lspci -v | grep -i vga
```