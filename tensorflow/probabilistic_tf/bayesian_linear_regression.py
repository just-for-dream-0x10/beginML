import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

"""
# Bayesian Linear Regression

This script implements Bayesian Linear Regression using TensorFlow Probability.
Bayesian Linear Regression extends traditional linear regression by:
1. Incorporating prior knowledge about parameters
2. Estimating full posterior distributions instead of point estimates
3. Providing uncertainty quantification for predictions
4. Naturally handling overfitting through regularization

The implementation uses variational inference to approximate the posterior
distributions of model parameters (weights and biases).
"""

tfd = tfp.distributions


# Generate synthetic data
def generate_data(n_samples=100, noise_std=0.5):
    np.random.seed(42)
    x = np.random.uniform(-1, 1, size=n_samples).astype(np.float32)
    true_w, true_b = 0.7, -0.5
    y = (
        true_w * x
        + true_b
        + np.random.normal(0, noise_std, size=n_samples).astype(np.float32)
    )
    return x, y, true_w, true_b


x_data, y_data, true_w, true_b = generate_data()
# Ensure data is tensor
x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)


# Define model - using a simpler approach
def bayesian_linear_regression():
    # Prior distributions
    w_prior = tfd.Normal(loc=0.0, scale=1.0)
    b_prior = tfd.Normal(loc=0.0, scale=1.0)

    # Variational posteriors
    w_loc = tf.Variable(0.0, name="w_loc")
    w_scale = tf.Variable(1.0, name="w_scale", constraint=lambda x: tf.nn.softplus(x))
    w_posterior = tfd.Normal(loc=w_loc, scale=w_scale)

    b_loc = tf.Variable(0.0, name="b_loc")
    b_scale = tf.Variable(1.0, name="b_scale", constraint=lambda x: tf.nn.softplus(x))
    b_posterior = tfd.Normal(loc=b_loc, scale=b_scale)

    return (
        (w_prior, b_prior),
        (w_posterior, b_posterior),
        [w_loc, w_scale, b_loc, b_scale],
    )


# Get prior and posterior distributions
priors, posteriors, variational_params = bayesian_linear_regression()
w_prior, b_prior = priors
w_posterior, b_posterior = posteriors


# Define loss function (ELBO)
@tf.function
def elbo():
    # Sample from posterior distributions
    w_samples = w_posterior.sample(10)
    b_samples = b_posterior.sample(10)

    # Calculate likelihood
    predictions = tf.reshape(w_samples, [-1, 1]) * x_data + tf.reshape(
        b_samples, [-1, 1]
    )
    log_likelihood = tf.reduce_mean(
        tfd.Normal(loc=predictions, scale=0.5).log_prob(y_data)
    )

    # Calculate KL divergence
    kl_w = tfd.kl_divergence(w_posterior, w_prior)
    kl_b = tfd.kl_divergence(b_posterior, b_prior)

    # Return ELBO (we want to maximize ELBO, so minimize -ELBO during training)
    return log_likelihood - (kl_w + kl_b)


# Train model
optimizer = tf.optimizers.Adam(learning_rate=0.1)


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = -elbo()  # Minimize negative ELBO
    gradients = tape.gradient(loss, variational_params)
    optimizer.apply_gradients(zip(gradients, variational_params))
    return loss


for i in range(1000):
    loss_value = train_step()
    if i % 200 == 0:
        print(f"Step {i}: ELBO = {-loss_value.numpy()}")

# Extract learned parameters
w_loc, w_scale, b_loc, b_scale = [p.numpy() for p in variational_params]
print(f"Learned weight: mean = {w_loc}, std = {w_scale}")
print(f"Learned bias: mean = {b_loc}, std = {b_scale}")
print(f"True weight: {true_w}, true bias: {true_b}")

# Visualize results
plt.figure(figsize=(10, 6))

# Set font for English text
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True

plt.scatter(x_data, y_data, alpha=0.5, label="Observed Data")

# Plot regression line
x_test = np.linspace(-1, 1, 100).astype(np.float32)
y_pred_mean = w_loc * x_test + b_loc

# Plot uncertainty intervals
num_samples = 100
w_samples = w_posterior.sample(num_samples).numpy()
b_samples = b_posterior.sample(num_samples).numpy()

y_samples = np.zeros((num_samples, len(x_test)))
for i in range(num_samples):
    y_samples[i, :] = w_samples[i] * x_test + b_samples[i]

y_mean = np.mean(y_samples, axis=0)
y_std = np.std(y_samples, axis=0)

plt.plot(x_test, y_pred_mean, "r-", label="Predicted Mean")
plt.fill_between(
    x_test,
    y_mean - 2 * y_std,
    y_mean + 2 * y_std,
    color="r",
    alpha=0.2,
    label="95% Confidence Interval",
)
plt.plot(x_test, true_w * x_test + true_b, "g--", label="True Function")
plt.legend()
plt.title("Bayesian Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("bayesian_linear_regression.png")
plt.show()
