import torch
import numpy as np

# Basic Tensor Creation
print("\n=== Basic Tensor Creation ===")
# Create tensor from different sources
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)
print(f"Tensor from numpy: {tensor_from_np}")

# Create tensor directly
direct_tensor = torch.tensor([1, 2, 3])
print(f"Direct tensor: {direct_tensor}")

# Create tensor with specific data type
tensor_float = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Float32 tensor: {tensor_float}")

# Create tensor with random values
random_tensor = torch.rand(2, 3)
print(f"Random tensor: {random_tensor}")

# Create tensor with zeros/ones
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(2, 3)
print(f"Zeros tensor: {zeros_tensor}")
print(f"Ones tensor: {ones_tensor}")

# Operations
print("\n=== Tensor Operations ===")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Element-wise operations
print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")

# Matrix operations
matrix_a = torch.rand(2, 3)
matrix_b = torch.rand(3, 2)
print(f"Matrix multiplication: {torch.matmul(matrix_a, matrix_b)}")
print(f"Element-wise multiplication: {matrix_a * matrix_b.T}")

# Broadcasting
broadcast_tensor = torch.tensor([1, 2, 3]) + 10
print(f"Broadcasting: {broadcast_tensor}")

# Indexing and Slicing
print("\n=== Indexing and Slicing ===")
tensor = torch.arange(12).reshape(3, 4)
print(f"Original tensor:\n{tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")

# Reshaping and Views
print("\n=== Reshaping and Views ===")
tensor = torch.arange(12)
print(f"Original tensor: {tensor}")
print(f"Reshape: {tensor.reshape(3, 4)}")
print(f"View (same memory): {tensor.view(3, 4)}")

# Concatenation and Stacking
print("\n=== Concatenation and Stacking ===")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(f"Concatenate along rows: {torch.cat([a, b], dim=0)}")
print(f"Concatenate along columns: {torch.cat([a, b], dim=1)}")
print(f"Stack along new dimension: {torch.stack([a, b])}")

# Advanced Operations
print("\n=== Advanced Operations ===")
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
print(f"Original tensor:\n{tensor}")
print(f"Sum: {tensor.sum()}")
print(f"Mean: {tensor.mean()}")
print(f"Max value: {tensor.max()}")
print(f"Max value and index: {tensor.max(dim=0)}")

# Gradients
print("\n=== Gradient Computation ===")
weights = torch.ones(2, 2, requires_grad=True)
print(f"Initial weights: {weights}")

# Forward pass
output = (weights * 3).sum()
print(f"Output: {output}")

# Backward pass
output.backward()
print(f"Gradients: {weights.grad}")

# Gradient accumulation
weights.grad.zero_()  # Reset gradients
output = (weights * 2).sum()
output.backward()
print(f"New gradients: {weights.grad}")

# Device handling
print("\n=== Device Handling ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move tensor to GPU
tensor = torch.ones(2, 2)
tensor = tensor.to(device)
print(f"Tensor on device: {tensor.device}")

# Memory Management
print("\n=== Memory Management ===")
# In-place operations
a = torch.tensor([1, 2, 3])
a.add_(1)  # In-place addition
print(f"In-place addition: {a}")

# Copy operations
b = a.clone()  # Creates a copy
print(f"Clone: {b}")

# Memory pinning for faster transfer
if torch.cuda.is_available():
    pinned_tensor = torch.ones(2, 2, pin_memory=True)
    print(f"Pinned tensor: {pinned_tensor.is_pinned()}")
