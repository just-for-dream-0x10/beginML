# TensorFlow Tensor Basic Concepts and Operations

## Table of Contents
- [1. Basic Concepts](#1-basic-concepts)
  - [1.1 What is a Tensor](#11-what-is-a-tensor)
  - [1.2 Key Attributes of Tensors](#12-key-attributes-of-tensors)
  - [1.3 Tensors of Different Ranks](#13-tensors-of-different-ranks)
  - [1.4 Relationship between Tensors and NumPy Arrays](#14-relationship-between-tensors-and-numpy-arrays)
- [2. Creating Tensors](#2-creating-tensors)
  - [2.1 Creating from Python Objects](#21-creating-from-python-objects)
  - [2.2 Creating Special Tensors](#22-creating-special-tensors)
  - [2.3 Random Tensors](#23-random-tensors)
  - [2.4 Creating from Other Data Sources](#24-creating-from-other-data-sources)
  - [2.5 Creating Mutable Tensors](#25-creating-mutable-tensors)
- [3. Tensor Operations](#3-tensor-operations)
  - [3.1 Basic Operations](#31-basic-operations)
  - [3.2 Shape Operations](#32-shape-operations)
  - [3.3 Indexing and Slicing](#33-indexing-and-slicing)
  - [3.4 Mathematical Operations](#34-mathematical-operations)
  - [3.5 Aggregation Operations](#35-aggregation-operations)
  - [3.6 Concatenation and Stacking](#36-concatenation-and-stacking)
  - [3.7 Copying and Tiling](#37-copying-and-tiling)
  - [3.8 Variable Operations](#38-variable-operations)
- [4. Broadcasting Mechanism](#4-broadcasting-mechanism)
  - [4.1 Examples of Broadcasting](#41-examples-of-broadcasting)
- [5. Advanced Operations](#5-advanced-operations)
  - [5.1 Conditional Operations](#51-conditional-operations)
  - [5.2 Gradient Computation](#52-gradient-computation)
  - [5.3 Custom Operations](#53-custom-operations)
- [6. Examples](#6-examples)
  - [6.1 Tensor <===> NumPy](#61-tensor-numpy)
  - [6.2 Device Placement](#62-device-placement)
  - [6.3 Memory Optimization](#63-memory-optimization)
- [7. Common Errors](#7-common-errors)
  - [7.1 Shape Mismatch](#71-shape-mismatch)
  - [7.2 Data Type Mismatch](#72-data-type-mismatch)
  - [7.3 Index Out of Bounds](#73-index-out-of-bounds)
  - [7.4 Vanishing/Exploding Gradients](#74-vanishingexploding-gradients)
  - [7.5 Out of Memory](#75-out-of-memory)
- [8. Advanced Applications](#8-advanced-applications)
  - [8.1 Custom Layers/Models](#81-custom-layersmodels)
  - [8.2 Advanced Indexing and Slicing](#82-advanced-indexing-and-slicing)
  - [8.3 Custom Training Loops](#83-custom-training-loops)
- [9. Performance Optimization Tips](#9-performance-optimization-tips)
  - [9.1 Speed Up with tf.function](#91-speed-up-with-tffunction)
  - [9.2 Data Loading Optimization](#92-data-loading-optimization)
  - [9.3 XLA Compilation](#93-xla-compilation)
- [10. Tensor Visualization](#10-tensor-visualization)
- [11. Application Examples](#11-application-examples)
  - [11.1 Image Processing](#111-image-processing)
  - [11.2 Natural Language Processing](#112-natural-language-processing)
  - [11.3 Time Series Processing](#113-time-series-processing)
  - [11.4 Model Training and Inference](#114-model-training-and-inference)
- [12. Interoperability with Other Frameworks](#12-interoperability-with-other-frameworks)
- [13. Summary and Best Practices](#13-summary-and-best-practices)
  - [13.1 Best Practices for Tensor Operations](#131-best-practices-for-tensor-operations)
  - [13.2 Common Pitfalls and Avoidance Methods](#132-common-pitfalls-and-avoidance-methods)
  - [13.3 Performance Optimization Checklist](#133-performance-optimization-checklist)

## 1. Basic Concepts

### 1.1 What is a Tensor

A tensor is the basic data structure in TensorFlow, which can be thought of as a multi-dimensional array. Mathematically, a tensor is a generalization of vectors and matrices to potentially higher dimensions.

In TensorFlow, tensors have the following characteristics:
- Fixed data type (e.g., `float32`, `int32`, etc.)
- Fixed shape (can be statically known or dynamically determined)
- Can run on different devices like CPU, GPU, or TPU
- Typically immutable (unless using `tf.Variable`)

### 1.2 Key Attributes of Tensors

- **Shape**: Indicates the number of elements in each dimension of the tensor
  - Example: `shape=(2,3,4)` represents a 3D tensor with dimensions 2×3×4
  - Shape can include `None`, indicating a dynamic size for that dimension
  - An empty tuple `()` represents a scalar (0D tensor)
  
- **Rank**: Indicates the number of dimensions of the tensor
  - Also known as `ndim` (number of dimensions)
  - Scalars have rank=0, vectors have rank=1, matrices have rank=2, and so on
  - Can be obtained using `tf.rank(tensor)` or `tensor.ndim`
  
- **Axis**: Indicates a specific dimension of the tensor
  - Counting starts from 0
  - Example: In a 3D tensor, axis=0 is the first dimension, axis=1 is the second, and axis=2 is the third
  - Negative indices can be used, such as axis=-1 for the last dimension
  
- **Size**: Indicates the total number of elements in the tensor
  - Equals the product of the sizes of all dimensions
  - Example: A tensor with shape=(2,3,4) has size=2×3×4=24
  - Can be obtained using `tf.size(tensor)`

- **Dtype**: The data type of the elements in the tensor
  - Common types: `tf.float32`, `tf.int32`, `tf.bool`, etc.
  - Full list includes: `float16`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`, `string`, `complex64`, `complex128`, etc.
  - Can be obtained using `tensor.dtype`

### 1.3 Tensors of Different Ranks

| Type | Rank | Example | Shape | Application |
|------|------|---------|-------|-------------|
| Scalar | 0 | `tf.constant(5)` | `()` | Single value, e.g., loss value |
| Vector | 1 | `tf.constant([1,2,3])` | `(3,)` | 1D features, e.g., time series |
| Matrix | 2 | `tf.constant([[1,2],[3,4]])` | `(2,2)` | 2D data, e.g., grayscale images |
| 3D Tensor | 3 | `tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])` | `(2,2,2)` | Color images (height, width, channels) |
| 4D Tensor | 4 | Batch image data | `(batch,height,width,channels)` | Batch processing of images |
| 5D Tensor | 5 | Video data | `(batch,frames,height,width,channels)` | Video sequences |

### 1.4 Relationship between Tensors and NumPy Arrays

TensorFlow tensors and NumPy arrays have many similarities but also important differences:

- **Similarities**:
  - Both support multi-dimensional array operations
  - Similar syntax (indexing, slicing, etc.)
  - Support broadcasting
  - Can be converted to each other

- **Differences**:
  - TensorFlow tensors support acceleration on GPU/TPU
  - TensorFlow tensors support automatic differentiation
  - TensorFlow tensors can be part of a computation graph
  - TensorFlow tensors are immutable by default (unless they are Variables)

## 2. Creating Tensors

### 2.1 Creating from Python Objects

```python
# Create a scalar
scalar = tf.constant(42)
print(scalar)  # tf.Tensor(42, shape=(), dtype=int32)

# Create a vector
vector = tf.constant([1, 2, 3, 4])
print(vector)  # tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)

# Create a matrix
matrix = tf.constant([[1, 2], [3, 4]])
print(matrix)  # tf.Tensor([[1 2] [3 4]], shape=(2, 2), dtype=int32)

# Create a tensor with a specific data type
float_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
print(float_tensor)  # tf.Tensor([[1. 2.] [3. 4.]], shape=(2, 2), dtype=float32)

# Create a string tensor
string_tensor = tf.constant(["Hello", "TensorFlow"])
print(string_tensor)  # tf.Tensor([b'Hello' b'TensorFlow'], shape=(2,), dtype=string)
```


### 2.2 Creating Special Tensors
```python
# Create a tensor of all zeros
zeros = tf.zeros(shape=(3, 4))
print(zeros)  # Tensor of shape (3, 4) filled with zeros

# Create a tensor of all ones
ones = tf.ones(shape=(2, 3, 4))
print(ones)  # Tensor of shape (2, 3, 4) filled with ones

# Create an identity matrix
identity = tf.eye(5)  # 5x5 identity matrix
print(identity)

# Create a diagonal matrix
diagonal = tf.linalg.diag([1, 2, 3, 4])  # Diagonal matrix with diagonal [1,2,3,4]
print(diagonal)

# Create a sequence of specified range
range_tensor = tf.range(start=0, limit=10, delta=2)  # [0, 2, 4, 6, 8]
print(range_tensor)

# Create a tensor filled with a specific value
fill_tensor = tf.fill([2, 3], 5)  # 2x3 tensor filled with 5
print(fill_tensor)

# Create a tensor with the same shape as another tensor
original = tf.constant([[1, 2], [3, 4]])
zeros_like = tf.zeros_like(original)  # Tensor of zeros with the same shape as original
ones_like = tf.ones_like(original)    # Tensor of ones with the same shape as original
print(zeros_like)
print(ones_like)
```


### 2.3 Random Tensors

```python
# Create a random tensor with normal distribution
random_normal = tf.random.normal(
    shape=(3, 2),
    mean=0.0,      # Mean
    stddev=1.0,    # Standard deviation
    dtype=tf.float32
)
print(random_normal)

# Create a random tensor with truncated normal distribution (limited to two standard deviations from the mean)
truncated_normal = tf.random.truncated_normal(
    shape=(3, 2),
    mean=0.0,
    stddev=1.0
)
print(truncated_normal)

# Create a random tensor with uniform distribution
random_uniform = tf.random.uniform(
    shape=(3, 2),
    minval=0,     # Minimum value
    maxval=10,    # Maximum value
    dtype=tf.float32
)
print(random_uniform)

# Use a seed to ensure reproducible random results
tf.random.set_seed(42)  # Set global seed
random_1 = tf.random.normal(shape=(3, 2))

tf.random.set_seed(42)  # Reset the same seed
random_2 = tf.random.normal(shape=(3, 2))

print("Are the random tensors equal:", tf.reduce_all(tf.equal(random_1, random_2)))  # True

# Use a generator object
generator = tf.random.Generator.from_seed(42)
random_tensor_1 = generator.normal(shape=(3, 2))
random_tensor_2 = generator.normal(shape=(3, 2))  # Different random values

# Recreate the same generator
generator_same = tf.random.Generator.from_seed(42)
random_tensor_3 = generator_same.normal(shape=(3, 2))  # Same as random_tensor_1

print("Is the first output of generator 1 and generator 2 equal:", 
      tf.reduce_all(tf.equal(random_tensor_1, random_tensor_3)))  # True
```


### 2.4 Creating from Other Data Sources
```python
import numpy as np

# Create a tensor from a NumPy array
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_np = tf.constant(numpy_array)
print(tensor_from_np)

# Preserve the data type of the NumPy array
float_array = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
tensor_from_float = tf.constant(float_array)
print(tensor_from_float.dtype)  # tf.float64

# Create a variable (mutable tensor) from a NumPy array
variable_tensor = tf.Variable(numpy_array)
print(variable_tensor)

# Direct conversion
tensor_converted = tf.convert_to_tensor(numpy_array)
print(tensor_converted)
```

### 2.5 Creating Mutable Tensors
```python
# Create a variable
var_1 = tf.Variable([1, 2, 3], dtype=tf.float32)
print(var_1)

# Create a variable from an existing tensor
tensor = tf.constant([[1, 2], [3, 4]])
var_2 = tf.Variable(tensor)
print(var_2)

# Create a variable with a specific name (useful when saving models)
named_var = tf.Variable([1, 2, 3], name="my_variable")
print(named_var)

# Create a variable with an initial value
zeros_var = tf.Variable(tf.zeros([2, 3]))
print(zeros_var)
```


## 3 Tensor Operations
### 3.1 Basic Operations
```python
# Get tensor attributes
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print("Shape:", tensor.shape)      # Shape: (2, 3)
print("Data type:", tensor.dtype)   # Data type: <dtype: 'int32'>
print("Rank:", tf.rank(tensor))    # Rank: 2
print("Number of dimensions:", tensor.ndim)     # Number of dimensions: 2
print("Total number of elements:", tf.size(tensor))  # Total number of elements: 6

# Type conversion
float_tensor = tf.cast(tensor, dtype=tf.float32)
print(float_tensor)

# Convert to NumPy array
numpy_array = tensor.numpy()  # or np.array(tensor)
print(numpy_array)

# Get the value of a tensor
scalar = tf.constant(42)
scalar_value = scalar.numpy()  # or int(scalar)
print(scalar_value)  # 42
```

### 3.2 Shape Operations
```python

# Change tensor shape
tensor = tf.constant([1, 2, 3, 4, 5, 6])
reshaped = tf.reshape(tensor, [2, 3])  # Reshape to a 2x3 matrix
print(reshaped)  # [[1 2 3] [4 5 6]]

# Automatically calculate dimension size (using -1)
auto_reshaped = tf.reshape(tensor, [3, -1])  # 3 rows, columns automatically calculated
print(auto_reshaped)  # [[1 2] [3 4] [5 6]]

# Flatten a tensor
matrix = tf.constant([[1, 2], [3, 4]])
flattened = tf.reshape(matrix, [-1])  # or tf.reshape(matrix, [4])
print(flattened)  # [1 2 3 4]

# Increase dimensions
expanded_0 = tf.expand_dims(tensor, axis=0)  # Add a dimension at axis 0
print(expanded_0.shape)  # (1, 6)

expanded_1 = tf.expand_dims(tensor, axis=1)  # Add a dimension at axis 1
print(expanded_1.shape)  # (6, 1)

# Increase dimensions using tf.newaxis
with_new_axis_start = tensor[tf.newaxis, ...]  # Add a dimension at the start
print(with_new_axis_start.shape)  # (1, 6)

with_new_axis_end = tensor[..., tf.newaxis]  # Add a dimension at the end
print(with_new_axis_end.shape)  # (6, 1)

# Remove dimensions of size 1
squeezed = tf.squeeze(expanded_0)  # Remove all dimensions of size 1
print(squeezed.shape)  # (6,)

# Remove specific dimensions
specific_squeeze = tf.squeeze(expanded_0, axis=0)  # Remove only dimension 0
print(specific_squeeze.shape)  # (6,)

# Transpose operation
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
transposed = tf.transpose(matrix)
print(transposed)  # [[1 4] [2 5] [3 6]]
```

##### Advanced Shape Operations

###### Multi-dimensional Transpose
```python
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```
###### Swap Dimension 0 and Dimension 2
```python
perm_tensor = tf.transpose(tensor_3d, perm=[2, 1, 0])
print(perm_tensor.shape)  # (2, 2, 2)
```
###### Adjust Tensor Shape to Match Another Tensor
```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic Indexing
print(tensor[0, 1])      # tf.Tensor(2, shape=(), dtype=int32) (First row, second column)
print(tensor[1])         # tf.Tensor([4 5 6], shape=(3,), dtype=int32) (Second row)

# Slicing [start:stop:step]
print(tensor[0:2, 1:3])  # tf.Tensor([[2 3] [5 6]], shape=(2, 2), dtype=int32) (First two rows, second and third columns)
print(tensor[:, 1])      # tf.Tensor([2 5 8], shape=(3,), dtype=int32) (Second column of all rows)
print(tensor[::2])       # tf.Tensor([[1 2 3] [7 8 9]], shape=(2, 3), dtype=int32) (Every other row)

# Using Ellipsis
print(tensor[..., 1])    # tf.Tensor([2 5 8], shape=(3,), dtype=int32) (Second column of all rows)

# Advanced Indexing
indices = tf.constant([[0, 1], [2, 2]])  # Index pairs (0,1) and (2,2)
gathered = tf.gather_nd(tensor, indices)
print(gathered)  # tf.Tensor([2 9], shape=(2,), dtype=int32)

# Boolean Mask Indexing
mask = tf.constant([[True, False, True], [False, True, False], [True, False, True]])
masked = tf.boolean_mask(tensor, mask)
print(masked)  # tf.Tensor([1 3 5 7 9], shape=(5,), dtype=int32)

# Using tf.slice
sliced = tf.slice(tensor, begin=[0, 1], size=[2, 2])
print(sliced)  # tf.Tensor([[2 3] [5 6]], shape=(2, 2), dtype=int32)

# Using tf.strided_slice for more complex slicing
strided = tf.strided_slice(tensor, [0, 0], [3, 3], [2, 2])  # Take every 2 steps
print(strided)  # tf.Tensor([[1 3] [7 9]], shape=(2, 2), dtype=int32)
```

### 3.4 Mathematical Operations
```python
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Basic Arithmetic
add = a + b  # or tf.add(a, b)
print("Addition:", add)  # [[6 8] [10 12]]

sub = a - b  # or tf.subtract(a, b)
print("Subtraction:", sub)  # [[-4 -4] [-4 -4]]

mul = a * b  # or tf.multiply(a, b) - Element-wise multiplication
print("Element-wise Multiplication:", mul)  # [[5 12] [21 32]]

div = a / b  # or tf.divide(a, b)
print("Division:", div)  # [[0.2 0.33333334] [0.42857143 0.5]]

# Matrix Multiplication
matmul = tf.matmul(a, b)  # or a @ b
print("Matrix Multiplication:", matmul)  # [[19 22] [43 50]]

# Dot Product
c = tf.constant([1, 2, 3])
d = tf.constant([4, 5, 6])
dot = tf.tensordot(c, d, axes=1)
print("Dot Product:", dot)  # 32 (1*4 + 2*5 + 3*6)

# Power Operation
power = tf.pow(a, 2)  # or a**2
print("Power Operation:", power)  # [[1 4] [9 16]]

# Square Root
sqrt = tf.sqrt(a)
print("Square Root:", sqrt)  # [[1. 1.4142135] [1.7320508 2.]]

# Exponential and Logarithm
exp = tf.exp(a)
print("Exponential:", exp)  # [[2.7182817 7.389056] [20.085537 54.59815]]

log = tf.math.log(a)
print("Natural Logarithm:", log)  # [[0. 0.6931472] [1.0986123 1.3862944]]

# Trigonometric Functions
sin = tf.sin(a)
cos = tf.cos(a)
tan = tf.tan(a)
print("Sine:", sin)
print("Cosine:", cos)
print("Tangent:", tan)

# Rounding Operations
rounded = tf.round(tf.constant([1.1, 2.5, 3.9]))
print("Round:", rounded)  # [1. 2. 4.]

floor = tf.floor(tf.constant([1.1, 2.5, 3.9]))
print("Floor:", floor)  # [1. 2. 3.]

ceil = tf.ceil(tf.constant([1.1, 2.5, 3.9]))
print("Ceil:", ceil)  # [2. 3. 4.]

# Absolute Value
abs_val = tf.abs(tf.constant([-1, -2, 3]))
print("Absolute Value:", abs_val)  # [1 2 3]

# Sign Function
sign = tf.sign(tf.constant([-2, 0, 3]))
print("Sign Function:", sign)  # [-1  0  1]

# Maximum and Minimum
maximum = tf.maximum(a, b)
print("Element-wise Maximum:", maximum)  # [[5 6] [7 8]]

minimum = tf.minimum(a, b)
print("Element-wise Minimum:", minimum)  # [[1 2] [3 4]]

# Clipping Values
clipped = tf.clip_by_value(a, clip_value_min=2, clip_value_max=3)
print("Clipped:", clipped)  # [[2 2] [3 3]]
```

### 3.5 Aggregation Operations
```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

# Sum
sum_all = tf.reduce_sum(tensor)  # Sum of all elements
print("Sum of all elements:", sum_all)  # 21.0

sum_axis0 = tf.reduce_sum(tensor, axis=0)  # Sum by columns
print("Sum by columns:", sum_axis0)  # [5. 7. 9.]

sum_axis1 = tf.reduce_sum(tensor, axis=1)  # Sum by rows
print("Sum by rows:", sum_axis1)  # [6. 15.]

sum_keep_dims = tf.reduce_sum(tensor, axis=1, keepdims=True)  # Keep dimensions
print("Sum by rows with dimensions kept:", sum_keep_dims)  # [[6.] [15.]]

# Mean
mean = tf.reduce_mean(tensor)  # Mean of all elements
print("Mean of all elements:", mean)  # 3.5

mean_axis0 = tf.reduce_mean(tensor, axis=0)  # Mean by columns
print("Mean by columns:", mean_axis0)  # [2.5 3.5 4.5]

# Maximum and Minimum
max_val = tf.reduce_max(tensor)  # Maximum value
print("Maximum value:", max_val)  # 6.0

min_val = tf.reduce_min(tensor)  # Minimum value
print("Minimum value:", min_val)  # 1.0

# Product
prod = tf.reduce_prod(tensor)  # Product of all elements
print("Product of all elements:", prod)  # 720.0

# Variance and Standard Deviation
variance = tf.math.reduce_variance(tensor)
print("Variance:", variance)

stddev = tf.math.reduce_std(tensor)
print("Standard Deviation:", stddev)

# Any and All
any_true = tf.reduce_any(tensor > 3)  # Any element greater than 3
print("Any element greater than 3:", any_true)  # True

all_true = tf.reduce_all(tensor > 0)  # All elements greater than 0
print("All elements greater than 0:", all_true)  # True

# Cumulative Sum
cumsum_0 = tf.cumsum(tensor, axis=0)  # Cumulative sum by columns
print("Cumulative sum by columns:", cumsum_0)  # [[1. 2. 3.] [5. 7. 9.]]

cumsum_1 = tf.cumsum(tensor, axis=1)  # Cumulative sum by rows
print("Cumulative sum by rows:", cumsum_1)  # [[1. 3. 6.] [4. 9. 15.]]

# Cumulative Product
cumprod = tf.cumprod(tensor, axis=1)  # Cumulative product by rows
print("Cumulative product by rows:", cumprod)  # [[1. 2. 6.] [4. 20. 120.]]

# Find Index of Maximum/Minimum
argmax = tf.argmax(tensor, axis=1)  # Index of maximum value in each row
print("Index of maximum value in each row:", argmax)  # [2 2]

argmin = tf.argmin(tensor, axis=0)  # Index of minimum value in each column
print("Index of minimum value in each column:", argmin)  # [0 0 0]
```

### 3.6 Concatenation and Stacking
```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Concatenation (merge along existing dimensions)
concat_0 = tf.concat([a, b], axis=0)  # Concatenate by rows, result shape: (4, 2)
print("Concatenate by rows:", concat_0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

concat_1 = tf.concat([a, b], axis=1)  # Concatenate by columns, result shape: (2, 4)
print("Concatenate by columns:", concat_1)
# [[1 2 5 6]
#  [3 4 7 8]]

# Stacking (create new dimensions)
stack = tf.stack([a, b])  # Stack with new dimension, result shape: (2, 2, 2)
print("Stack:", stack)
# [[[1 2]
#   [3 4]]
#  [[5 6]
#   [7 8]]]

stack_0 = tf.stack([a, b], axis=0)  # Stack along dimension 0, equivalent to tf.stack([a, b])
print("Stack along dimension 0:", stack_0.shape)  # (2, 2, 2)

stack_1 = tf.stack([a, b], axis=1)  # Stack along dimension 1
print("Stack along dimension 1:", stack_1.shape)  # (2, 2, 2)

stack_2 = tf.stack([a, b], axis=2)  # Stack along dimension 2
print("Stack along dimension 2:", stack_2.shape)  # (2, 2, 2)

# Splitting
split_0 = tf.split(concat_0, num_or_size_splits=2, axis=0)  # Split by rows evenly
print("Split by rows evenly:", [s.shape for s in split_0])  # [(2, 2), (2, 2)]

split_1 = tf.split(concat_1, num_or_size_splits=[1, 3], axis=1)  # Split by columns unevenly
print("Split by columns unevenly:", [s.shape for s in split_1])  # [(2, 1), (2, 3)]

# Unstacking
unstacked = tf.unstack(stack, axis=0)  # Unstack along dimension 0
print("Unstack:", [u.shape for u in unstacked])  # [(2, 2), (2, 2)]
```


### 3.7 Copying and Tiling
```python
tensor = tf.constant([[1, 2], [3, 4]])

# Copy Tensor
tiled = tf.tile(tensor, [2, 3])  # Copy 2 times along the first dimension, 3 times along the second dimension
print("Tiled:", tiled)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]

# Copy Single Dimension
tiled_rows = tf.tile(tensor, [3, 1])  # Copy rows only
print("Copy rows:", tiled_rows)
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]
#  [1 2]
#  [3 4]]

tiled_cols = tf.tile(tensor, [1, 2])  # Copy columns only
print("Copy columns:", tiled_cols)
# [[1 2 1 2]
#  [3 4 3 4]]

# Copy Multi-dimensional Tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
tiled_3d = tf.tile(tensor_3d, [2, 1, 3])
print("3D Tensor Tiled:", tiled_3d.shape)  # (4, 2, 6)

# Broadcast Copy (more memory efficient)
broadcast = tf.broadcast_to
# Broadcast Copy (more memory efficient)
broadcast_tensor = tf.broadcast_to(tensor, [4, 2])  # Broadcast to shape (4, 2)
print("Broadcast:", broadcast_tensor)
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]]
```

### 3.8 Variable Operations
```python
# Create Variable
var = tf.Variable([1, 2, 3])

# Update Variable
var.assign([4, 5, 6])  # Completely replace values
print("After replacement:", var)  # [4 5 6]

var[0].assign(10)      # Update specific index
print("After index update:", var)  # [10 5 6]

var.assign_add([1, 1, 1])  # Add a value
print("After addition:", var)  # [11 6 7]

var.assign_sub([1, 0, 0])  # Subtract a value
print("After subtraction:", var)  # [10 6 7]

# Update Variable with Control Flow
condition = tf.constant(True)
var.assign(tf.where(condition, [1, 1, 1], var))  # Assign [1,1,1] if condition is true
print("After conditional update:", var)  # [1 1 1]

# Gradient Tracking for Variables
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x * x
grad = tape.gradient(y, x)
print("Gradient:", grad)  # 6.0 (dy/dx = 2x, when x=3 is 6)
```


## 4 Broadcasting Mechanism
TensorFlow supports broadcasting, allowing operations on tensors of different shapes as long as they are compatible in certain dimensions.
``` python
# Broadcasting Example
a = tf.constant([[1, 2, 3],
                 [4, 5, 6]])  # Shape: (2, 3)
b = tf.constant([10, 20, 30])  # Shape: (3,)

# b will be broadcast to shape (2, 3)
c = a + b  # Result: [[11, 22, 33], [14, 25, 36]]
print("Broadcast Addition:", c)

# Scalar Broadcasting
scalar = tf.constant(5)
d = a + scalar  # Scalar will be broadcast to all elements
print("Scalar Broadcasting:", d)  # [[6 7 8] [9 10 11]]

# Complex Broadcasting
e = tf.constant([[1], [2]])  # Shape: (2, 1)
f = tf.constant([10, 20, 30])  # Shape: (3,)
g = e + f  # e is broadcast to (2,3), f is broadcast to (2,3)
print("Complex Broadcasting:", g)
# [[11 21 31]
#  [12 22 32]]
```

Broadcasting Rules:

1. Match from the trailing dimensions
2. If a tensor has a dimension of 1, or the other tensor does not have this dimension, it can be broadcast
3. After broadcasting, the size of this dimension becomes the larger of the two tensors


#### 4.1 Examples of Broadcasting
```python
# Batch data normalization
batch_data = tf.random.normal([32, 10])  # 32 samples, each with 10 features
mean = tf.reduce_mean(batch_data, axis=0)  # Shape: (10,)
std = tf.math.reduce_std(batch_data, axis=0)  # Shape: (10,)
normalized = (batch_data - mean) / std  # Broadcasting applied to each sample
print("Shape after normalization:", normalized.shape)  # (32, 10)

# Add bias to the convolutional layer output
conv_output = tf.random.normal([32, 28, 28, 64])  # Batch size = 32, height = 28, width = 28, channels = 64
bias = tf.Variable(tf.zeros([64]))  # One bias per channel
biased_output = conv_output + bias  # bias is broadcast to shape (32, 28, 28, 64)
print("Shape after adding bias:", biased_output.shape)  # (32, 28, 28, 64)

# Batch matrix multiplication
batch_a = tf.random.normal([32, 10, 20])  # 32 matrices of 10x20
single_b = tf.random.normal([20, 30])  # One 20x30 matrix
batch_result = tf.matmul(batch_a, single_b)  # single_b is broadcast, result shape (32, 10, 30)
print("Shape of batch matrix multiplication result:", batch_result.shape)  # (32, 10, 30)    
```

## 5 Advanced Operations

### 5.1 Conditional Operations
```python
x = tf.constant(2)
y = tf.constant(5)

# Conditional selection
result = tf.where(x > y, x, y)  # Returns x if x > y, otherwise returns y
print("Conditional selection:", result)  # 5

# Complex conditions
tensor = tf.constant([[1, 2], [3, 4]])
mask = tensor > 2
masked_tensor = tf.boolean_mask(tensor, mask)  # [3, 4]
print("Boolean mask result:", masked_tensor)

# Conditional execution
def f1(): return tf.constant(10)
def f2(): return tf.constant(20)
result = tf.cond(tf.less(x, y), f1, f2)  # Executes f1 if x < y, otherwise executes f2
print("Conditional execution:", result)  # 10

# Complex conditional logic
values = tf.constant([1, 2, 3, 4, 5, 6])
even_mask = tf.equal(values % 2, 0)
odd_mask = tf.logical_not(even_mask)
even_values = tf.boolean_mask(values, even_mask)  # [2, 4, 6]
odd_values = tf.boolean_mask(values, odd_mask)    # [1, 3, 5]
print("Even numbers:", even_values)
print("Odd numbers:", odd_values)
```

### 5.2 Gradient Computation
```python
# Basic gradient computation
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x
    
# Compute dy/dx
dy_dx = tape.gradient(y, x)  # Result: 6.0
print("dy/dx =", dy_dx)

# Multi-variable gradients
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x1**2 + 2*x1*x2 + x2**3
    
gradients = tape.gradient(y, [x1, x2])
print("∂y/∂x1 =", gradients[0])  # 2*x1 + 2*x2 = 2*2 + 2*3 = 10
print("∂y/∂x2 =", gradients[1])  # 2*x1 + 3*x2^2 = 2*2 + 3*3^2 = 31

# Second-order derivatives
x = tf.Variable(3.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x**3
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)
print("d²y/dx² =", d2y_dx2)  # 6*x = 6*3 = 18

# Stop gradient propagation
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    # Stop gradient propagation for y
    y_stopped = tf.stop_gradient(y)
    z = x**2 + y_stopped**2
    
gradients = tape.gradient(z, [x, y])
print("∂z/∂x =", gradients[0])  # 2*x = 2*2 = 4
print("∂z/∂y =", gradients[1])  # 0 (because gradient propagation is stopped)
```


### 5.3 Custom Operations
```python
# Define a function for graph execution using the tf.function decorator
@tf.function
def custom_operation(x, y):
    return tf.sqrt(x) + tf.square(y)

result = custom_operation(tf.constant(4.0), tf.constant(3.0))
print("Custom operation result:", result)  # 2.0 + 9.0 = 11.0

# Custom operation with conditional logic
@tf.function
def abs_if_positive(x, y):
    if tf.reduce_sum(x) > 0:
        return tf.abs(y)
    else:
        return y

result = abs_if_positive(tf.constant([1, 2]), tf.constant([-5, -6]))
print("Conditional custom operation:", result)  # [5, 6]

# Custom gradient
@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
        return dy * (1 - 1/(1 + e))
    return tf.math.log(1 + e), grad

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = log1pexp(x)
dy_dx = tape.gradient(y, x)
print("Custom gradient:", dy_dx)
```


## 6 example
### 6.1 TENSOR <===> Numpy
```python
# Tensor to NumPy
tensor = tf.constant([[1, 2], [3, 4]])
numpy_array = tensor.numpy()  # or np.array(tensor)
print("Converted to NumPy:", numpy_array)
print("Type:", type(numpy_array))  # <class 'numpy.ndarray'>

# NumPy to Tensor
array = np.array([[5, 6], [7, 8]])
tensor = tf.convert_to_tensor(array)
print("Converted to Tensor:", tensor)
print("Type:", type(tensor))  # <class 'tensorflow.python.framework.ops.EagerTensor'>

# Preserve data type
float_array = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
float_tensor = tf.convert_to_tensor(float_array)
print("Preserved data type:", float_tensor.dtype)  # tf.float64

# Shared memory (in some cases)
tensor = tf.constant([1, 2, 3])
numpy_array = tensor.numpy()
numpy_array[0] = 100  # Modify NumPy array
print("Modified NumPy array:", numpy_array)  # [100, 2, 3]
print("Original tensor:", tensor)  # [1, 2, 3] (unaffected because constant is immutable)

# Shared memory with variables
var_tensor = tf.Variable([1, 2, 3])
var_numpy = var_tensor.numpy()
var_numpy[0] = 100  # Modify NumPy array
print("Modified NumPy array:", var_numpy)  # [100, 2, 3]
print("Original variable:", var_tensor)  # [1, 2, 3] (unaffected because numpy() creates a copy)

# Update variable from NumPy
var_tensor.assign(var_numpy)  # Update variable using NumPy array
print("Updated variable:", var_tensor)  # [100, 2, 3]
```

### 6.2 Device Placement
```python
# Specify device
with tf.device('/CPU:0'):
    cpu_tensor = tf.constant([[1, 2], [3, 4]])
    print("CPU tensor:", cpu_tensor)

# If GPU is available
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        gpu_tensor = tf.constant([[1, 2], [3, 4]])
        print("GPU tensor:", gpu_tensor)

# Check the device of a tensor
print("Device:", cpu_tensor.device)  # /job:localhost/replica:0/task:0/device:CPU:0

# Cross-device operations (TensorFlow handles data transfer automatically)
if tf.config.list_physical_devices('GPU'):
    with tf.device('/CPU:0'):
        a = tf.constant([[1, 2], [3, 4]])
    with tf.device('/GPU:0'):
        b = tf.constant([[5, 6], [7, 8]])
    c = a + b  # TensorFlow handles cross-device operations
    print("Cross-device result:", c)
    print("Result device:", c.device)  # Usually on GPU, as TensorFlow prefers faster devices

# Force placement on a specific device
if tf.config.list_physical_devices('GPU'):
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    with tf.device('/CPU:0'):
        c = a + b  # Force execution on CPU
        print("Forced CPU result device:", c.device)

```

### 6.3 Memory Optimization
```python
# Use tf.function for graph optimization
@tf.function
def compute_intensive_function(x):
    result = tf.constant(0.0)
    for i in range(1000):
        result += tf.reduce_sum(tf.square(x))
    return result

# Graph execution is more efficient than eager execution
x = tf.random.normal([1000, 1000])
result = compute_intensive_function(x)
print("Computation result:", result)

# Control memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print("Error setting memory growth:", e)

# Use mixed precision
if tf.config.list_physical_devices('GPU'):
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

# Use tf.data for efficient data processing
dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([10000, 10]))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
for batch in dataset.take(1):
    print("Batch shape:", batch.shape)
```


## 7 Common Errors
### 7.1 Shape Mismatch

```python
# Problem example
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([5, 6, 7])
# c = a + b  # This will raise a shape mismatch error

# Solution 1: Adjust shape
b_reshaped = tf.reshape(b, [1, 3])
# Still mismatched, but at least we understand the problem

# Solution 2: Create compatible shape
b_compatible = tf.constant([5, 6])
c = a + b_compatible[:, tf.newaxis]  # Broadcast to shape (2, 2)
print("Resolved shape mismatch:", c)
```
solutions:

- Check tensor shapes for compatibility ( print(tensor.shape) )
- Use tf.reshape to adjust shapes
- Understand broadcasting rules
- Use tf.expand_dims or [..., tf.newaxis] to add dimensions

### 7.2 Data Type Mismatch
InvalidArgumentError: Cannot convert a tensor of type float to an intege
```python
# Problem example
a = tf.constant(1, dtype=tf.int32)
b = tf.constant(2.5, dtype=tf.float32)
# c = tf.bitwise.left_shift(a, b)  # This will raise a type error because bitwise operations require integers

# Solution: Convert data type
b_int = tf.cast(b, dtype=tf.int32)
c = tf.bitwise.left_shift(a, b_int)
print("Resolved type mismatch:", c)
```
Solutions:

- Use tf.cast to convert data types
- Check required data types for operations ( print(tensor.dtype) )
- Specify correct data types when creating tensors

### 7.3 Index Out of Bounds
```python
# Problem example
tensor = tf.constant([1, 2, 3])
# value = tensor[5]  # This will raise an index out of bounds error

# Solution 1: Use valid index
valid_index = 2
value = tensor[valid_index]
print("Valid index:", value)

# Solution 2: Use tf.clip_by_value to limit index
index = tf.constant(5)
safe_index = tf.clip_by_value(index, 0, tf.shape(tensor)[0] - 1)
value = tensor[safe_index]
print("Safe index:", value)
```
Solutions:

- Ensure index is within valid range
- Use tf.clip_by_value to limit index range
- Use tf.gather and tf.boolean_mask for safe indexing


### 7.4 Vanishing/Exploding Gradients
Problem: Gradients become NaN or infinite during training
```python
# Problem example
x = tf.Variable([1000.0])
with tf.GradientTape() as tape:
    y = tf.exp(x)  # Exponential of large values may cause gradient explosion
grad = tape.gradient(y, x)
print("Potential gradient explosion:", grad)

# Solution 1: Gradient clipping
x = tf.Variable([5.0])
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)
clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
print("Gradient before clipping:", grad)
print("Gradient after clipping:", clipped_grad)

# Solution 2: Use more stable functions
x = tf.Variable([100.0])
with tf.GradientTape() as tape:
    # Use tf.math.log1p and tf.math.expm1 instead of log and exp
    y = tf.math.log1p(x)  # log(1+x), numerically more stable
grad = tape.gradient(y, x)
print("Stable function gradient:", grad)
```
Solutions:

- Use gradient clipping ( tf.clip_by_value , tf.clip_by_norm )
- Use numerically stable functions ( tf.math.log1p , tf.math.expm1 )
- Appropriately scale input data
- Use batch normalization or layer normalization



### 7.5 Out of Memory
ResourceExhaustedError: OOM when allocating tensor
`ResourceExhaustedError: OOM when allocating tensor`
```python
# Problem: Creating overly large tensor
# large_tensor = tf.random.normal([50000, 50000])  # May cause OOM error

# Solution 1: Reduce batch size or model size
smaller_tensor = tf.random.normal([5000, 5000])  # More reasonable size

# Solution 2: Use tf.data for batching
dataset = tf.data.Dataset.from_tensor_slices(
    tf.random.normal([10000, 1000])).batch(32)
for batch in dataset.take(1):
    print("Batch size:", batch.shape)

# Solution 3: Enable memory growth
# See previous GPU memory growth settings
```
Solutions:

- Reduce batch size or model size
- Use tf.data for batching
- Enable GPU memory growth
- Use mixed precision training
- Optimize model architecture


## 8  Advanced Applications
### 8.1 Custom Layers/Models
```python
# Custom layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MyDenseLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Use custom layer
inputs = tf.random.normal([32, 10])
layer = MyDenseLayer(5)
outputs = layer(inputs)
print("Custom layer output shape:", outputs.shape)  # (32, 5)
```


### 8.2 Advanced Indexing and Slicing
```python
# Use tf.gather_nd for advanced indexing
params = tf.reshape(tf.range(1, 13), [3, 4])
print("Parameter tensor:\n", params)

indices = tf.constant([[0, 0], [1, 2], [2, 3]])
gathered = tf.gather_nd(params, indices)
print("gather_nd result:", gathered)  # [1, 7, 12]

# Use tf.scatter_nd for advanced index updates
indices = tf.constant([[0, 0], [1, 2]])
updates = tf.constant([100, 200])
shape = tf.constant([3, 4])
scattered = tf.scatter_nd(indices, updates, shape)
print("scatter_nd result:\n", scattered)

# Use tf.boolean_mask for mask indexing
tensor = tf.reshape(tf.range(1, 13), [3, 4])
mask = tf.constant([[True, False, False, True],
                    [False, True, True, False],
                    [True, False, True, False]])
masked = tf.boolean_mask(tensor, mask)
print("boolean_mask result:", masked)  # [1, 4, 6, 7, 9, 11]
```

### 8.3 Custom Training Loops
```python
# Define model, loss function, and optimizer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

# Custom training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Generate example data
x = tf.random.normal([32, 5])
y = tf.random.normal([32, 1])

# Execute training step
loss = train_step(x, y)
print("Training loss:", loss.numpy())
```


## 9 Performance Optimization Tips
### 9.1 Speed Up with tf.function
```python
# Function without tf.function
def slow_function(x):
    result = tf.constant(0.)
    for i in range(100):
        result += tf.reduce_sum(tf.square(x))
    return result

# Function with tf.function
@tf.function
def fast_function(x):
    result = tf.constant(0.)
    for i in range(100):
        result += tf.reduce_sum(tf.square(x))
    return result

# Compare performance
x = tf.random.normal([1000, 1000])

import time
start = time.time()
slow_result = slow_function(x)
slow_time = time.time() - start

start = time.time()
fast_result = fast_function(x)
fast_time = time.time() - start

print(f"Without tf.function: {slow_time:.4f} seconds")
print(f"With tf.function: {fast_time:.4f} seconds")
print(f"Speedup: {slow_time/fast_time:.2f}x")

```

#### Principles

1. Graph Execution vs Eager Execution
TensorFlow has two execution modes:
- Eager Execution: The default mode, where Python code is executed line by line.
- Graph Execution: The entire computation process is compiled into a computation graph and then executed.

The `@tf.function` decorator converts Python functions into TensorFlow computation graphs, enabling the transition from eager execution to graph execution.

2. Global Optimization
Computation graphs allow TensorFlow to perform global optimizations, such as:
- Operation fusion
- Common subexpression elimination
- Constant folding
- Memory optimization

3. Parallel Execution
Computation graphs can automatically analyze dependencies between operations and execute independent operations in parallel.

4. Better Hardware Utilization
Graph execution can better utilize hardware accelerators like GPUs/TPUs, reducing data transfer between CPU and accelerators.

Functions decorated with `@tf.function` are 2-10 times faster than regular functions, and the speedup can be even higher for compute-intensive operations with loops.

Tips:
While `@tf.function` can provide significant performance improvements, there are some considerations:

1. Tracing Overhead: There is an initial tracing overhead during the first execution, making it suitable for functions that are called repeatedly.
2. Python Side Effects: In graph execution mode, Python print statements, file operations, and other side effects may not execute as expected.
3. Dynamic Control Flow: Conditional statements based on TensorFlow tensor values require special handling using `tf.cond` and similar functions.
4. Increased Debugging Complexity: Debugging in graph execution mode is more complex than in eager execution.

In summary, `@tf.function` converts Python code into optimized computation graphs, reducing Python interpreter overhead and achieving global optimization, thereby significantly improving execution speed.


### ## 9.2 Data Loading Optimization
```python
# Create a sample dataset
dataset = tf.data.Dataset.from_tensor_slices(
    tf.random.normal([10000, 32]))

# Unoptimized dataset
unoptimized = dataset.batch(32)

# Optimized dataset
optimized = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
optimized = optimized.cache()  # Cache data
optimized = optimized.map(
    lambda x: x * 2, 
    num_parallel_calls=tf.data.AUTOTUNE  # Parallel mapping
)
# In practical applications, these optimizations can significantly improve performance
```


### 9.3 XLA compilation
```python
# Enable XLA compilation
@tf.function(jit_compile=True)
def xla_function(x):
    return tf.reduce_sum(tf.square(x))

# Without XLA
@tf.function
def normal_function(x):
    return tf.reduce_sum(tf.square(x))

# Compare performance
x = tf.random.normal([1000, 1000])

import time
start = time.time()
normal_result = normal_function(x)
normal_time = time.time() - start

start = time.time()
xla_result = xla_function(x)
xla_time = time.time() - start

print(f"Without XLA: {normal_time:.4f} seconds")
print(f"With XLA: {xla_time:.4f} seconds")
print(f"Speedup: {normal_time/xla_time:.2f}x")
```

## 10 Tensor Visualization
```python
# Visualize tensors using TensorBoard
import datetime

# Create log directory
log_dir = "logs/tensor_demo/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Record scalars
with writer.as_default():
    for i in range(100):
        tf.summary.scalar('sine_wave', tf.sin(i/10), step=i)

# Record images
image = tf.random.normal([100, 100, 3])  # Create random image
image = tf.clip_by_value(image, 0, 1)  # Clip to [0,1] range
with writer.as_default():
    tf.summary.image("random_image", [image], step=0)

# Record histograms
with writer.as_default():
    for i in range(5):
        values = tf.random.normal([1000], mean=i, stddev=1)
        tf.summary.histogram("distribution", values, step=i)

# Record text
with writer.as_default():
    tf.summary.text("markdown_text", 
                   "# TensorFlow Visualization\nThis is a **Markdown** formatted text.", 
                   step=0)

print("You can start TensorBoard with the following command:")
print(f"tensorboard --logdir={log_dir}")
```

## 11 Application Examples
### 11.1 Image Processing
```python
# Load and preprocess images
def load_and_preprocess_image(path):
    # Read image file
    img = tf.io.read_file(path)
    # Decode to RGB image
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize
    img = tf.image.resize(img, [224, 224])
    # Normalize to [0,1]
    img = img / 255.0
    return img

# Image augmentation
def augment_image(image):
    # Random flip
    image = tf.image.random_flip_left_right(image)
    # Random brightness adjustment
    image = tf.image.random_brightness(image, 0.2)
    # Random contrast adjustment
    image = tf.image.random_contrast(image, 0.8, 1.2)
    # Random crop
    image = tf.image.random_crop(image, [200, 200, 3])
    # Resize back to original size
    image = tf.image.resize(image, [224, 224])
    return image

# Batch process images
def process_image_batch(image_paths, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

### 11.2 Natural Language Processing
```python
# Text tokenization and encoding
def tokenize_and_encode(texts, vocab_size=10000, max_length=100):
    # Create tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    # Convert text to integer sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to make them the same length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post')
    
    return padded_sequences, tokenizer

# Use TensorFlow's text processing features
def process_text_with_tf(texts):
    # Create vocabulary
    vocab = tf.keras.layers.TextVectorization(max_tokens=10000)
    vocab.adapt(texts)
    
    # Convert text to vectors
    vectorized_text = vocab(texts)
    return vectorized_text, vocab
```

### 11.3 Time Series Processing
```python
# Create time series windows
def create_time_series_windows(data, window_size=10, batch_size=32):
    # Create windowed dataset
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=data[window_size:],
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=True
    )
    return dataset

# Process time series with TensorFlow
def process_time_series(data, window_size=10):
    # Convert data to tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    
    # Create inputs and targets
    x = data_tensor[:-1]  # All data points except the last one
    y = data_tensor[1:]   # All data points except the first one
    
    # Create windows
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x_window, y_window: 
                              tf.data.Dataset.zip((
                                  x_window.batch(window_size),
                                  y_window.batch(window_size)
                              )))
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset
```

### 11.4  Model Training and Inference
```python
# Create a simple neural network model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Custom training loop
def custom_training_loop(model, train_dataset, epochs=10):
    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # Training step
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_accuracy(y, predictions)
        return loss
    
    # Training loop
    for epoch in range(epochs):
        # Reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for x_batch, y_batch in train_dataset:
            loss = train_step(x_batch, y_batch)
            
        # Print progress
        print(f'Epoch {epoch+1}, Loss: {train_loss.result():.4f}, '
              f'Accuracy: {train_accuracy.result():.4f}')
    
    return model

# Model inference
def model_inference(model, input_data):
    # Preprocess input data
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    
    # Make predictions
    predictions = model(input_tensor, training=False)
    
    # Get most likely classes
    predicted_classes = tf.argmax(predictions, axis=1)
    
    # Get probabilities
    probabilities = tf.reduce_max(tf.nn.softmax(predictions, axis=1), axis=1)
    
    return predicted_classes, probabilities
```

## 12 ## Interoperability with Other Frameworks
```python
# Requires PyTorch installation: pip install torch
import torch
import numpy as np

# Convert TensorFlow tensor to PyTorch tensor
def tf_to_torch(tf_tensor):
    # Convert to NumPy array first
    np_array = tf_tensor.numpy()
    # Then convert to PyTorch tensor
    torch_tensor = torch.from_numpy(np_array)
    return torch_tensor

# Convert PyTorch tensor to TensorFlow tensor
def torch_to_tf(torch_tensor):
    # Convert to NumPy array first
    np_array = torch_tensor.detach().cpu().numpy()
    # Then convert to TensorFlow tensor
    tf_tensor = tf.convert_to_tensor(np_array)
    return tf_tensor

# Example
tf_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
torch_tensor = tf_to_torch(tf_tensor)
print("PyTorch tensor:", torch_tensor)

back_to_tf = torch_to_tf(torch_tensor)
print("Converted back to TensorFlow:", back_to_tf)
```

## 13. Summary and Best Practices
### 13.1 Best Practices for Tensor Operations
1. Use vectorized operations: Avoid Python loops whenever possible, and use TensorFlow's vectorized operations (e.g., tf.map_fn ).
2. Utilize broadcasting: Understand and leverage broadcasting to reduce memory usage and improve performance.
3. Ensure data type consistency: Make sure tensors involved in operations have compatible data types to avoid unnecessary type conversions.
4. Use tf.function : Accelerate execution of compute-intensive functions by using the @tf.function decorator.
5. Batch processing: Use batch processing instead of single sample processing, especially when training models.
6. Memory management:
   - Use tf.data for batching large tensors
   - Set large intermediate tensors to None for garbage collection
   - Use mixed precision training to reduce memory footprint
7. Device placement: Be aware of which device tensors are on to avoid unnecessary data transfers between devices.
### 13.2 Common Pitfalls and Avoidance Methods
1. Shape mismatch: Check tensor shapes before operations and use print(tensor.shape) for debugging.
2. Vanishing/exploding gradients: Use gradient clipping, batch normalization, and appropriate activation functions.
3. Memory leaks: Be cautious when creating tensors in loops, especially within tf.function .
4. Excessive CPU/GPU switching: Try to keep related operations on the same device.
5. Ignoring data types: Explicitly specify data types, especially when dealing with floating-point numbers.
### 13.3 Performance Optimization Checklist
- Use tf.function to decorate compute-intensive functions
- Use tf.data for efficient data loading and preprocessing
- Enable XLA compilation for accelerated computation
- Use mixed precision training (for supported GPUs)
- Optimize batch size to balance memory usage and training speed
- Use tf.TensorArray instead of Python lists for accumulating results in loops
- Use tf.config.experimental.set_memory_growth to control GPU memory growth
- Use tf.distribute for distributed training

