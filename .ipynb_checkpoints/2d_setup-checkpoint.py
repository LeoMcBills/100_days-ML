'''With the help of init_device_mesh(), we can accomplish the above 2D setup in just two lines, and we can still access the underlyng ProcessGroup if needed.'''
from torch.distributed.device_mesh import init_device_mesh
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))

# Users can access the underlying process group through `get_group` API.
replicate_group = mesh_2d.get_group(mesh_dim="replicate")
shard_group = mesh_2d.get_group(mesh_dim="shard")