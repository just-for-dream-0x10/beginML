## Regularization Techniques in Tensorflow

Regularization is used to improve the generalization of machine learning models by preventing complex models from overfitting the training data. Below are some common regularization techniques, along with an introduction to the concepts of overfitting and underfitting.

### Understanding Overfitting and Underfitting
- Overfitting: Occurs when a model learns the training data (including its noise and outliers) too well, resulting in poor performance on unseen data. This happens when the model is too complex relative to the amount of data and noise.
- Underfitting: Occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and unseen data.
In summary, overfitting is characterized by excellent performance on training data but poor generalization to new data, while underfitting is characterized by poor performance on both training and test data due to the model's inability to learn the patterns in the data. Regularization techniques aim to find a balance between these two extremes, allowing the model to generalize well to new data.

### L1 Regularization (Lasso)
Adds a penalty equal to the absolute value of the coefficients. It can lead to sparse models with some feature weights being zero, representing model complexity as the sum of the absolute values of the weights.
$$ 
L1_loss = \lambda \sum |w_i|
$$
```python
model.add(tf.keras.layers.Dense(units=128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01)))
```

### L2 Regularization (Ridge)
Adds a penalty equal to the square of the coefficients. It helps reduce model complexity and prevent overfitting. Model complexity is represented as the sum of the squares of the weights.
$$
L2_loss = \lambda \sum w_i^2
$$

```python
model.add(tf.keras.layers.Dense(units=128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
```

### ElasticNet Regularization
Combines L1 and L2 regularization penalties. It is useful when you want to gain the benefits of both L1 and L2 regularization.
$$
ElasticNet_loss = \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2
$$

```python
model.add(tf.keras.layers.Dense(units=128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
```


### Dropout
Randomly sets a small portion of input units to zero during each update in training, which helps prevent overfitting.

```python
model.add(tf.keras.layers.Dropout(rate=0.5))
```

### Early Stopping
Stops training when the validation loss starts to increase, preventing the model from learning noise in the training data.
```python
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

### Data Augmentation
Increases the diversity of training data by applying random transformations such as rotation, shifting, and flipping.

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])
```


### Batch Normalization
Normalizes the input of each layer, which helps stabilize learning and reduce overfitting.
```python
model.add(tf.keras.layers.BatchNormalization())
```
