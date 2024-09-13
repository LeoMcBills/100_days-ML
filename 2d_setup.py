'''With the help of init_device_mesh(), we can accomplish the above 2D setup in just two lines, and we can still access the underlyng ProcessGroup if needed.'''
from torch.distributed.device_mesh import init_device_mesh
mesh_2d = init_device_mesh("cuda", ())