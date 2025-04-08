import tensorflow as tf
import numpy as np
import time

print(f"version :{tf.__version__}")

sclar = tf.constant(100)
print("sclar : \n", sclar, type(sclar), sclar.ndim, "\n")
vector = tf.constant([1, 2, 3, 4])
print("vector: \n", vector, vector.ndim, "\n")
matrix = tf.constant([[10, 7], [7, 10]])
print("matrix: \n", matrix.ndim, matrix, "\n")

another_matrix = tf.constant(
    [[10.0, 7.0], [3.0, 2.0], [8.0, 9.0]], dtype=tf.float16
)  # specify the data type with dtype parameter

print("another_matrix: \n", another_matrix.ndim, another_matrix, "\n")

tensor = tf.constant(
    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]
)
print("tensor : \n", tensor.ndim, tensor, "\n")

#  mutable and unmutable tensor
change_tensor = tf.Variable([1, 2, 3])
unmut_tensor = tf.constant([1, 2, 3])
print("mut tensor && numut tensor \n", change_tensor, unmut_tensor)

change_tensor[0].assign(90)
change_tensor.assign_add([1, 2, 3])
print("mut tensor: ", change_tensor, "\n")

# create random tensor
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3, 2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))
print(
    "random tensor : \n", random_1, random_2, "\n", random_1.shape, random_2.shape, "\n"
)
assert tf.reduce_all(tf.equal(random_1, random_2))

# Shuffle a tensor (valuable for when you want to shuffle your data)
# If both global and operation seeds are set: The two seeds are combined to determine the random sequence.
# "tf.random.set_seed(42)" sets the global seed, and the seed parameter in "tf.random.shuffle(seed=42)" sets the operation seed.
# This is because "operations that rely on a random seed actually derive it from two seeds: the global and operation-level seeds. This sets the global seed."
print(tf.random.shuffle(another_matrix))
print(tf.random.shuffle(another_matrix, seed=42))

# Make a tensor of all ones
tf.ones(shape=(3, 2))

# Make a tensor of all zeros
tf.zeros(shape=(3, 2))

# create tensor form numpy array
np_tensor = tf.constant(np.array([3, 7, 10]))
print(np_tensor, np_tensor.ndim)

np_arr = np.arange(1, 25, dtype=np.int32).reshape(2, 3, 4)
tensor_from_np = tf.Variable(np_arr, dtype=tf.int32)

print(
    "np from tensor :\n ",
    np_arr,
    "\n",
    f"{tensor_from_np}\n{tensor_from_np.shape}",
    tf.rank(tensor_from_np),
)

# create tensor with all zeros shape(2,3,4,5)
# aixs0= 2, axis1=3, axis2=4, axis3=5
zeros_tensor = tf.zeros([2, 3, 4, 5])
print(f"Shape: {zeros_tensor.shape}")
print(f"Number of dimensions: {zeros_tensor.ndim}")
print(f"Total number of elements: {tf.size(zeros_tensor)}")

# create tensor with all zeros shape(2,3,4,5,1)
# aixs0= 2, axis1=3, axis2=4, axis3=5, axis4=1
zeros_tensor_1 = zeros_tensor[..., tf.newaxis]
print(f"zeros_tensor_1 Shape: {zeros_tensor_1.shape}")
print(f"Number of dimensions: {zeros_tensor_1.ndim}")
print(f"Total number of elements: {tf.size(zeros_tensor_1)}")
print(zeros_tensor_1)

base_ones = tf.ones([2, 3])
# axis is the dimension index
new_base_ones = tf.expand_dims(base_ones, axis=2)
print(new_base_ones, "\n shape: ", new_base_ones.shape, "\n")

tiled_tensor = tf.tile(new_base_ones, [1, 1, 2])
multiplier = tf.constant([5, 10], dtype=tf.float32)
multiplier_reshaped = tf.reshape(multiplier, [1, 1, 2])

result_tensor = tiled_tensor * multiplier_reshaped
print("result tensor : ", result_tensor, "\n shape: ", result_tensor.shape, "\n")


def slow_function(x):
    result = tf.constant(0.0)
    for i in range(1000):
        result += tf.reduce_sum(tf.square(x))
    return result


@tf.function
def fast_function(x):
    return tf.reduce_sum(tf.square(x)) * 1000


x = tf.random.normal([1_000, 1_000])

start = time.time()
slow_result = slow_function(x)
slow_time = time.time() - start

start = time.time()
fast_result = fast_function(x)
fast_time = time.time() - start

print(f"use tf.function: {slow_time:.4f} seconds")
print(f"without tf.function: {fast_time:.4f} seconds")
print(f"improvement: {slow_time/fast_time:.2f}x")

print("Available CPU", tf.config.list_physical_devices("CPU"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

if hasattr(tf.config, "list_physical_devices") and hasattr(tf, "config"):
    try:
        print("Available MPS devices:", tf.config.list_physical_devices("MPS"))
    except:
        print("MPS device not found or not supported in this TensorFlow version")


diagonal = tf.linalg.diag([1, 2, 3, 4])
print("Diagonal matrix:", diagonal)
