'''
INFORMATION from PyTorch Fundamentals: https://www.learnpytorch.io/00_pytorch_fundamentals/
TENSORS

[colour_channels, height, width] = [3, 224, 224]
- three colour channles, and 224 pixels in height and width

torch.Tensor : multi-dimensional matrix containing elements of a single data type


'''
# tensor can be constructed from Python list or sequence

import torch
import numpy as np

tensor1 = torch.tensor([[1., -1.], [1., -1.]]) #python list
tensor2 = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]])) #numpy array

print(tensor1)
print(tensor2)

# tensor of specific data can be constructed (shape, data type, where it lives):

torch.zeros([2, 4], dtype=torch.int32) # tensor of zeros
# cuda0 = torch.device('cuda:0') # use GPU if available (the first NVIDIA GPU in your computer)
torch.ones([2, 4], dtype=torch.float64) # tensor of 1.0000

# when using in .py files,use the following to see info

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[1][2]) # prints 6 ( line 1, column 2), (zeros is part of the lines and columns)
x[0][1] = 8 # change 2 to 8
print(x)

# use torch.Tensor.item() to get a Python number from a tensor containing a single value

x = torch.tensor([[1]])
print(x) # prints : tensor([[ 1]])
print(x.item()) # prints 1

# SCALAR: single number
scalar = torch.tensor(7)
print(scalar) # prints : tensor(7)
print(scalar.ndim) # check dimensions (0 for scalar)
print(scalar.item())  # get the Python number within a tensor (only works with one-element tensors)

# VECTOR = single dimension tensor
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim) # dimension 1
vector.shape #torch.Size([2]) (2 elements in the vector)

# MATRIX = two-dimensional tensor
MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX)
print(MATRIX.ndim) # dimension 2
print(MATRIX.shape) # torch.Size([2, 2]) (2 rows, 2 columns)

# TENSOR 
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim) # dimension 3
print(TENSOR.shape) # torch.Size([1, 3, 3]) (3 rows, 3 columns)
# there is 1 dimesion of 3 by 3

#----------------------------------------------------------------------------------

# RANDOM TENSORS  

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(random_tensor.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3)) #[height, width, color_channels]
print(random_image_size_tensor)
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros.dtype)

#Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)

#----------------------------------------------------------------------------------

# RANGE AND TENSORS LIKE

# torch.arange(start, end, step)
# start = start of range (e.g. 0)
# end = end of range (e.g. 10)
# step = how many steps in between each value (e.g. 1)

# Note: In Python, you can use range() to create a range. However in PyTorch, torch.range() is deprecated and may show an error in the future.

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)

#----------------------------------------------------------------------------------

# Tensor Datatypes

# torch.cuda = tensor being used for GPU
