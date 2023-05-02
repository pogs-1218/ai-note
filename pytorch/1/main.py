import torch
import numpy as np

data = [[1, 2], [3, 4]]
print(type(data))

# copied? yes!
# data type
# requires_grad
x_data = torch.tensor(data)
print(type(x_data))
print(x_data)


np_data = np.array([[1, 2], [3, 4]])
print(type(np_data))

# share the same memory
# what if I create a tensor in GPU device?
#x_np = torch.from_numpy(np_data)

# as_tensor has device parameter!
# is it copied in that device is not CPU?
x_np = torch.as_tensor(np_data, device=torch.device('cuda'))

np.multiply(np_data, 2, out=np_data)
print(f'numpy multiply:\n {np_data}')
print(f'tensor multiply:\n {x_np}')

x_ones = torch.ones_like(x_np)
print(f'ones tensor:\n {x_ones}')
x_rand = torch.rand_like(x_np, dtype=torch.float)
print(f'rand tensor:\n {x_rand}')

# uniform distribution [0,1)
#rand_tensor = torch.rand(2, 3, device=torch.device('cuda'))
rand_tensor = torch.rand(3, 3)
print(rand_tensor)
print(f'shape: {rand_tensor.shape}')
print(f'device: {rand_tensor.device}')
print(f'is cuda: {rand_tensor.is_cuda}')
print(f'dim: {rand_tensor.ndim}')


if torch.cuda.is_available():
  rand_tensor_cuda = rand_tensor.to('cuda')
  print(f'cuda rand:\n{rand_tensor_cuda}')
  # is it copied? or moved? let's check.
  rand_tensor_cuda[0,0] = -0.1111
  print('-----------------')
  print(f'{rand_tensor}')
  print(f'{rand_tensor_cuda}')

t1 = torch.ones(3, 2)
t2 = torch.zeros(3, 2)
new_t = torch.cat([t1, t2], dim=1)
print(f'{new_t}')
