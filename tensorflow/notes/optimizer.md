# TensorFlow Optimizers

Optimizers are algorithms used in deep learning to minimize the loss function. They adjust model parameters to reduce the difference between predicted and actual values.

 [1. Basic Optimizers](#1-basic-optimizers)
  - [1.1 SGD (Stochastic Gradient Descent)](#11-sgd-stochastic-gradient-descent)
  - [1.2 SGD with Momentum](#12-sgd-with-momentum)
- [2. Adaptive Learning Rate Optimizers](#2-adaptive-learning-rate-optimizers)
  - [2.1 AdaGrad (Adaptive Gradient Algorithm)](#21-adagrad-adaptive-gradient-algorithm)
  - [2.2 RMSprop (Root Mean Square Propagation)](#22-rmsprop-root-mean-square-propagation)
  - [2.3 Adam (Adaptive Moment Estimation)](#23-adam-adaptive-moment-estimation)
  - [2.4 AdamW (Adam with Weight Decay)](#24-adamw-adam-with-weight-decay)
- [3. Optimizer Comparison](#3-optimizer-comparison)
  - [3.1 Performance Comparison](#31-performance-comparison)
  - [3.2 Selection Recommendations](#32-selection-recommendations)
- [4. Optimizer Hyperparameter Tuning](#4-optimizer-hyperparameter-tuning)
  - [4.1 Learning Rate Tuning](#41-learning-rate-tuning)
  - [4.2 Other Hyperparameter Tuning](#42-other-hyperparameter-tuning)
- [5. Examples](#5-examples)
  - [5.1 Natural Language Processing](#51-natural-language-processing)
  - [5.2 Image Classification](#52-image-classification)
- [6. Common Issues and Solutions](#6-common-issues-and-solutions)
  - [6.1 Gradient Explosion/Vanishing](#61-gradient-explosionvanishing)
  - [6.2 Learning Rate Too High/Low](#62-learning-rate-too-highlow)
  - [6.3 Overfitting](#63-overfitting)
- [7. Optimizer Visualization and Comparison](#7-optimizer-visualization-and-comparison)
  - [7.1 Optimizer Convergence Speed Comparison](#71-optimizer-convergence-speed-comparison)
- [8. Advanced Optimization Techniques](#8-advanced-optimization-techniques)
  - [8.1 Learning Rate Warmup](#81-learning-rate-warmup)
  - [8.2 Cyclical Learning Rate](#82-cyclical-learning-rate)
  - [8.3 Mixed Precision Training](#83-mixed-precision-training)
- [9. Summary](#9-summary)
  - [9.1 Optimizer Selection Recommendations](#91-optimizer-selection-recommendations)
  - [9.2 Hyperparameter Tuning Recommendations](#92-hyperparameter-tuning-recommendations)
  - [9.3 Best Practices](#93-best-practices)

## 1. Basic Optimizers

### 1.1 SGD (Stochastic Gradient Descent)

**Basic Principle:**
- SGD is the most basic optimizer. It calculates the gradient of the loss function with respect to the model parameters and updates the parameters in the opposite direction of the gradient.
- It is called "stochastic" because each iteration uses only one or a few samples to compute the gradient.

**Mathematical Expression:**
$$ θ = θ - η * ∇J(θ) $$
where θ is the model parameter, η is the learning rate, and ∇J(θ) is the gradient of the loss function with respect to the parameter.

**Advantages:**
- Simple and easy to implement.
- May converge to a better local minimum in some problems.
- Low memory requirements.

**Disadvantages:**
- Slow convergence and prone to oscillation in ravines.
- Performance is highly sensitive to the choice of learning rate.
- Uses the same learning rate for all parameters, which may not be suitable for all situations.

**TensorFlow Implementation:**
```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
```

### 1.2 SGD with Momentum
Basic principle:

- In SGD, we add momentum to accelerate convergence and reduce oscillation.
- Momentum item records the previous gradient's "direction", helping the optimization process to be smoother.
Mathematical expression:
$$ 
v = γ * v + η * ∇J(θ) \\
θ = θ - v
$$
where v is the velocity vector, γ is the momentum coefficient (usually set to 0.9).

Positive points:

- Accelerate convergence, especially when high curvature, small but consistent gradient direction.
- Reduce oscillation, making the training process more stable.
Negative points:

- Introduces an extra hyperparameter (momentum coefficient) that needs to be adjusted.
- In some cases, it may "overshoot" the most optimal points.

TensproFlow implementation:
```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
```

## 2. Adaptive Learning Rate Optimizers
### 2.1 AdaGrad (Adaptive Gradient Algorithm)

Basic Principle:

- Adjusts the learning rate adaptively for each parameter.
- Uses a smaller learning rate for frequently updated parameters and a larger learning rate for infrequently updated parameters.
Mathematical Expression: 
$$
G = G + (∇J(θ))² \
θ = θ - η * ∇J(θ) / √(G + ε)
$$
where G is the cumulative sum of squared gradients, and ε is a small constant to prevent division by zero.

Advantages:

- Automatically adjusts the learning rate without manual tuning.
- Suitable for handling sparse data.
Disadvantages:

- The learning rate monotonically decreases, which may lead to premature stopping of training.
- May perform poorly in deep learning.
TensorFlow Implementation:

TensorFlow implementation:
```python
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1)
```



### 2.2 RMSprop

Basic principle:
- RMSprop is an adaptive learning rate optimizer. It adjusts the learning rate for each parameter based on the gradient square's exponential moving average.
- For gradient larger parameters, reduce learning rate; for gradient smaller parameters, increase learning rate.
- Resolved the problem of AdaGrad learning rate monotonically decreasing.

Mathematical formula:
$$
E[g²] = ρ * E[g²] + (1-ρ) * (∇J(θ))² \\
θ = θ - η * ∇J(θ) / √(E[g²] + ε)
$$


where E[g²] is the gradient square's exponential moving average, ρ is the decay rate (usually set to 0.9).

Positive points:

- Adaptive learning rate, insensitive to the learning rate choice.
- Good performance on non-stationary target functions.
- Suitable for sparse gradient.
Negative points:

- Still dependent on the global learning rate.
- May not converge to the best solution.

Tensorflow implementation
```python
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0)
```

### 2.3 Adam (Adaptive Moment Estimation)

Basic Principle:

- Adam combines the advantages of RMSprop and momentum methods.
- It maintains an exponentially moving average of both the squared gradients and the gradients (momentum).
- Adjusts the learning rate for each parameter adaptively by considering both the first and second moments of the gradients.

Mathematical Expression: 
$$
m = β₁ * m + (1-β₁) * ∇J(θ)    \text{{(First moment - momentum)}}\
v = β₂ * v + (1-β₂) * (∇J(θ))²  \text{{(Second moment)}} \
m̂ = m / (1-β₁ᵗ)                 \text{{(Bias correction)}} \
v̂ = v / (1-β₂ᵗ)                 \text{{(Bias correction)}} \
θ = θ - η * m̂ / (√v̂ + ε)
$$

where β₁ and β₂ are decay rates (usually set to 0.9 and 0.999, respectively), and t is the iteration number.

Advantages:

- Combines the advantages of RMSprop and momentum methods, with fast convergence and good performance.
- Adaptive learning rate, insensitive to the choice of learning rate.
- Performs well on various problems.
- Includes bias correction for more stable early iterations.
Disadvantages:

- Slightly higher computational complexity than SGD and RMSprop.
- May generalize worse than SGD in some cases.
- May lead to overfitting in some situations.


### 2.4 AdamW (Adam with Weight Decay)
Basic Principle:

- AdamW is a variant of Adam that correctly implements weight decay (instead of L2 regularization).
- Weight decay is applied directly to the weight updates, not through the gradients.
Advantages:

- Better generalization ability than Adam.
- Performs better on large models.
Disadvantages:

- Introduces an additional hyperparameter (weight decay rate) that needs tuning.
TensorFlow Implementation:
```python
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004)
```


## 3 Optimizer Comparison
### 3 Performance Comparison
|optimizer|Advantages|Disadvantages|Application Scenarios|
|---|---|---|---|
|SGD|	Simple and easy to implement, may converge to a better local minimum|	Slow convergence, sensitive to learning rate choice|Simple problems, low convergence speed requirements|
|SGD with Momentum|Faster convergence, can find global minimum faster|	Requires tuning of momentum parameter|	Scenarios requiring momentum|
|AdaGrad|Can find global minimum faster|Monotonically decreasing learning rate|Convex optimization problems, sparse data|
|RMSprop|Adaptive learning rate, insensitive to learning rate choice, suitable for sparse gradients|depends on global learning rate|Non-stationary objectives, sparse gradients|
|Adam|Combines advantages of RMSprop and momentum methods, fast convergence, good performance, adaptive learning rate|Slightly higher computational complexity|Most problems, especially complex ones|
|AdamW|Better generalization ability than Adam|Requires tuning of weight decay rate|Large models, need better generalization ability|

### 3.2  Selection Recommendations
- SGD: When you want interpretable results or are dealing with very large datasets where computational efficiency is important.
- SGD with Momentum: When SGD converges too slowly or oscillates too much.
- AdaGrad: When dealing with sparse features (e.g., text data).
- RMSprop: When training recurrent neural networks or dealing with non-stationary objectives.
- Adam: As a general-purpose optimizer, it usually works well for many problems.
- AdamW: When using large models and focusing on generalization ability.

## 4  Optimizer Hyperparameter Tuning
### 4.1 Learning Rate Tuning
The learning rate is one of the most important hyperparameters, affecting the convergence speed and performance of the model.

Common strategies:

- Grid Search: Try a range of learning rates (e.g., 0.1, 0.01, 0.001, 0.0001).
- Learning Rate Scheduling: Gradually decrease the learning rate as training progresses.
- Learning Rate Warmup: Start with a small learning rate and gradually increase to the target learning rate.
TensorFlow Learning Rate Scheduling Example:

```python
# xponential decay learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

### 4.2 Other Hyperparameter Tuning
- Momentum Coefficient: Usually set to 0.9.
- β₁ and β₂ (Adam): Usually set to 0.9 and 0.999, respectively.
- ε (Numerical Stability Constant): Usually set to 1e-7 or 1e-8.
- Weight Decay Rate: Usually between 0.0001 and 0.01.

## 5 Examples
### 5.1 Natural Language Processing
```python
# Train a Transformer model using the AdamW optimizer
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.004
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 5.2 Image Classification
```python
#  Train a CNN using the Adam optimizer
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## 6. Common Issues and Solutions
### 6.1 Gradient Explosion/Vanishing
Problem: Gradients become very large (explosion) or very small (vanishing) during backpropagation.

Solutions:

- Use Gradient Clipping.
- Use Batch Normalization.
- Use appropriate activation functions (e.g., ReLU).
- Use appropriate weight initialization methods.
TensorFlow Gradient Clipping Example:


```python
# TensorFlow Gradient Clipping Example
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.0)  # Gradient norm clipping
# Or
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, clipvalue=0.5)  # Gradient value clipping
```

### 6.2 Learning Rate Too High/Low
Problem: A learning rate that is too high may cause divergence, while a learning rate that is too low may lead to slow convergence or getting stuck in local minima.

Solutions:

- Use a learning rate scheduler.
- Use learning rate warmup.
- Use adaptive learning rate optimizers (e.g., Adam).

```python
# Use a learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=5, 
    min_lr=0.0001
)

model.fit(x_train, y_train, callbacks=[reduce_lr])
```
### 6.3  Overfitting
Problem: The model performs well on the training set but poorly on the validation set.

Solutions:

- Use regularization (L1/L2).
- Use Dropout.
- Use Early Stopping.
- Use Data Augmentation.
```python
# Use Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train, callbacks=[early_stopping])
```




##  7. ## Optimizer Visualization and Comparison
ifferent optimizers may behave very differently during the optimization process. Below are visual comparisons on some common optimization problems:
![op1](./images/opt1.gif)

![op2](./images/opt2.gif)

The visualizations illustrate the behavior of different optimization algorithms in challenging scenarios. In the first image, the saddle point visualization shows how SGD struggles to break symmetry and gets stuck at the top due to varying curvature across dimensions. In contrast, algorithms like RMSprop can detect very low gradients in the saddle point direction, increase the effective learning rate along that direction, and help RMSprop move forward. The second image depicts the contour of the loss surface and the temporal evolution of different optimization algorithms. Note the "overshooting" behavior of momentum-based methods, which makes optimization like a ball rolling down a hill.


### 7.1  Optimizer Convergence Speed Comparison
In practical applications, the convergence speed and final performance of different optimizers may vary significantly. Below is a simple comparison example:：
```python
# Compare the performance of different optimizers
optimizers = [
    tf.keras.optimizers.SGD(learning_rate=0.01),
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    tf.keras.optimizers.RMSprop(learning_rate=0.001),
    tf.keras.optimizers.Adam(learning_rate=0.001)
]

optimizer_names = ['SGD', 'SGD with Momentum', 'RMSprop', 'Adam']
histories = []

for i, optimizer in enumerate(optimizers):
    print(f"Training with {optimizer_names[i]}...")
    model = create_model()  # Create the same model architecture
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, verbose=0)
    histories.append(history)
    
# Plot learning curve comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=optimizer_names[i])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
for i, history in enumerate(histories):
    plt.plot(history.history['val_accuracy'], label=optimizer_names[i])
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

## 8 Advanced Optimization Techniques

### 8.1 Learning Rate Warmup
Learning rate warmup is a technique where a smaller learning rate is used initially during training, and then gradually increased to the target learning rate. This helps stabilize the initial training process.
```python   
# Learning Rate Warmup Example
class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, decay_schedule):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_schedule = decay_schedule
        
    def __call__(self, step):
        # Linearly increase learning rate during warmup phase
        warmup_lr = self.initial_learning_rate * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        # Use decay schedule after warmup
        decay_lr = self.decay_schedule(step - self.warmup_steps)
        # Use warmup or decay learning rate depending on current step
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)

# Create decay schedule
decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)

# Create learning rate schedule with warmup
warmup_lr = WarmupScheduler(
    initial_learning_rate=0.0001,  # Initial learning rate
    warmup_steps=1000,            # Warmup steps
    decay_schedule=decay_schedule  # Decay schedule after warmup
)

# Use learning rate schedule with warmup
optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_lr)
```

### 8.2 Cyclical Learning Rate
Cyclical learning rate is a technique where the learning rate is periodically changed during training, helping to escape local minima.

```python
# Cyclical Learning Rate Example
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, max_lr, step_size):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        
    def __call__(self, step):
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0., 1 - x)

# Create cyclical learning rate schedule
clr = CyclicalLearningRate(
    base_lr=0.0001,  # Minimum learning rate
    max_lr=0.001,    # Maximum learning rate
    step_size=2000   # Half cycle steps
)

# Use cyclical learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=clr)
```
### 8.3 ### Mixed Precision Training
Mixed precision training uses a mix of float16 and float32 precision to accelerate training while maintaining model accuracy.
```python

# Mixed Precision Training Example
from tensorflow.keras import mixed_precision

# Set global policy
mixed_precision.set_global_policy('mixed_float16')

# Create model
model = tf.keras.Sequential([
    # Model layers...
])

# Ensure output layer uses float32
model.add(tf.keras.layers.Activation('softmax', dtype='float32'))

# Use loss scaling to prevent gradient underflow
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## 9 Summary
Optimizers are crucial components in deep learning, and choosing the right optimizer and hyperparameters can significantly impact the training speed and performance of a model.

### 9.1 Optimizer Selection Recommendations
1. Initial Attempt: Start with Adam, as it performs well in most cases.
2. Need Better Generalization: Try SGD with Momentum or AdamW.
3. Limited Computational Resources: Use SGD or RMSprop.
4. Specific Tasks:
   - Computer Vision: Adam, SGD with Momentum
   - Natural Language Processing: Adam, AdamW
   - Reinforcement Learning: Adam, RMSprop
### 9.2 Hyperparameter Tuning Recommendations
1. Learning Rate: Often the most important hyperparameter, recommended to try [0.1, 0.01, 0.001, 0.0001].
2. Batch Size: Typically between 16-512, depending on available memory and dataset size.
3. Optimizer-Specific Parameters: Default values work in most cases, but may need adjustment for specific problems.
### 9.3 Best Practices
1. Monitor Training: Closely observe training and validation loss curves.
2. Use Callbacks: Utilize TensorFlow's callback mechanisms (e.g., early stopping, learning rate scheduling).
3. Regularization: Combine techniques like weight decay, dropout to prevent overfitting.
4. Gradient Clipping: Use gradient clipping for models prone to gradient explosion, like RNNs.
5. Learning Rate Scheduling: Use learning rate schedulers to improve final performance.