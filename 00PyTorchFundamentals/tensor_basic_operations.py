""" BASIC TENSOR OPERATIONS 
+ -> addition
- -> subtraction
* -> multiplication
/ -> division

"""



import torch
import numpy as np
from time import time

tensor1 = torch.tensor([1,2,3])
print(tensor1 + 10)
print(tensor1 * 10)

# the values inside the tensor will not change unless they are reassigned

# Example of how to reassign values:

tensor1 = tensor1 - 10
print(tensor1)

tensor1 = tensor1 + 10
print(tensor1)

# BUILT-IN PyTorch FUNCTIONS:

# torch.add() -> addition
# torch.subtract() -> subtraction
# torch.mul() -> multiplication
# torch.divide() -> division

print(torch.multiply(tensor1, 10))
print(tensor1) #original tensor1 values will not change unless reassigned

# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor1, "*", tensor1)
print("Equals:", tensor1 * tensor1)


""" MATRIX OPERATIONS

- torch.matmul() -> matrix multiplication

inner dimension must match! 
resulting matrix will have the outer dimensions
"@" in Python is the symbol for matrix multiplication

- (3,2) @ (3,2) cannot be multiplied
- (3,2) @ (2,3) can be multiplied ->  (3,3)
- (3,2) @ (2,4) can be multiplied ->  (3,4)

 """

tensor2 = torch.tensor([1,2,3]) 
print(tensor2.shape)

# Element-wise matrix multiplication
print(tensor2 * tensor2)
# Matrix multiplication
print(torch.matmul(tensor2, tensor2)) # (1,3) @ (3,1) -> (1,1)

# Matrix multiplication by hand 

start_time1 = time()
# Matrix multiplication by hand 
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(tensor2)):
  value += tensor2[i] * tensor2[i]
value
print(value)

end_time1 = time()
print(f"Time taken: {end_time1 - start_time1}")

start_time2= time()
torch.matmul(tensor2, tensor2)
end_time2 = time()
print(f"Time taken: {end_time2 - start_time2}")

# Counting better the time!!!!!!
from time import perf_counter

start_time1 = perf_counter()

# Manual operation
value = 0
for i in range(len(tensor2)):
    value += tensor2[i] * tensor2[i]

end_time1 = perf_counter()
print(f"Manual calculation: {end_time1 - start_time1:.10f} seconds")

start_time2 = perf_counter()

# PyTorch operation
result = torch.matmul(tensor2, tensor2)

end_time2 = perf_counter()
print(f"PyTorch calculation: {end_time2 - start_time2:.10f} seconds")

# ---------------------------------------------------------------------------

# SWITCH DIMESIONS OF TENSORS

# torch.transpose(input, dim0, dim1) (input is the desired tensor to transpose and dim0 and dim1 are the dimensions to be swapped)
# tensor.T - where tensor is the desired tensor to transpose

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# View tensor_A and tensor_B
print(tensor_A) #(3,2)
print(tensor_B) #(3,2)

# View tensor_A and tensor_B.T
print(tensor_A) #(3,2)
print(tensor_B.T) #(2,3) -> transposed tensor_B

# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")

# CAN ALSO USE torch.mm() for matrix multiplication (only works with 2D tensors)
print(torch.mm(tensor_A, tensor_B.T))

#-----------------------------------------------------------------------------------------

# y = x{A^T} +b -> linear regression model : torch.nn.Linear() 
# matrix multiplication between an input x and a weights matrix A.
# - x is the input to the layer
# - A is the weights matrix created by the layer, this starts out as random numbers that get adjusted as a neural network learns to better represent patterns in the data
# - notice the "T", that's because the weights matrix gets transposed
# - b is the bias term used to slightly offset the weights and inputs
# - y is the output (a manipulation of the input in the hopes to discover patterns in it)

# Since the linear layer starts with a random weights matrix, let's make it reproducible 

torch.manual_seed(42) # it has a fixed seed so you can : reproduce the same initial values when you run the same code under the same conditions

# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# FIND MIN, MAX, MEAN, SUM....

# Create a tensor
x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")

# can do the same with the following functions:
print(torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x))


# POSITIONAL MIN/MAX

# index of tesnor where max or min occur

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")


# CHANGE TENSOR DATATYPE : the lower the number the less precise the computer stores the value
# - the lower the number also the faster the operations

# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(f"Tensor dtype: {tensor.dtype}")
print(tensor)
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)
print(f"Tensor float16 dtype: {tensor_float16.dtype}")
# Create an int8 tensor
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)

"""
Reshaping, stacking, squeezing and unsqueezing (change dimensions without changing values inside them)

- torch.reshape(input, shape): Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
- Tensor.view(shape): Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
- torch.stack(tensors, dim=0): Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
- torch.squeeze(input): Squeezes input to remove all the dimenions with value 1.
- torch.unsqueeze(input, dim): Returns input with a dimension value of 1 added at dim.
- torch.permute(input, dims): Returns a view of the original input with its dimensions permuted (rearranged) to dims.

"""

# Create a tensor
x = torch.arange(1., 8.)
print(x, x.shape)

#Add extra dimension
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

# Change view (keeps same data as original but changes view)
z = x.view(1, 7)
print(z, z.shape)
# Changing the view changes the original tensor too
# Changing z changes x
z[:, 0] = 5
print(z, x)

# If we wanted to stack our new tensor on top of itself five times, we could do so with torch.stack().
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print(x_stacked)



# How about removing all single dimensions from a tensor?
#To do so you can use torch.squeeze() (I remember this as squeezing the tensor to only have dimensions over 1).

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# And to do the reverse of torch.squeeze() you can use torch.unsqueeze() to add a dimension value of 1 at a specific index.

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")