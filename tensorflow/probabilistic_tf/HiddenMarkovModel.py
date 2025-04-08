import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
# Hidden Markov Model Implementation

This script demonstrates the implementation and analysis of a Hidden Markov Model (HMM)
using TensorFlow Probability. HMMs are statistical models where:
1. The system being modeled follows a Markov process with unobservable (hidden) states
2. Each state has a probability distribution over possible output observations
3. State transitions follow first-order Markov dynamics

Key features demonstrated:
- Model definition with initial state, transition, and observation distributions
- Sampling from the model
- Computing sequence likelihood using the Forward algorithm
- Finding the most likely state sequence using the Viterbi algorithm
- Computing posterior marginal probabilities
- Visualizing and evaluating model performance
"""

# Set random seed for reproducibility
tf.random.set_seed(42)

# Create HMM model
hmm = tfp.distributions.HiddenMarkovModel(
    # Initial state distribution
    initial_distribution=tfp.distributions.Categorical(probs=[0.8, 0.2]),
    # Transition matrix
    transition_distribution=tfp.distributions.Categorical(
        probs=[[0.7, 0.3], [0.2, 0.8]]  # Transition probabilities for state 0
    ),  # Transition probabilities for state 1
    # Observation distribution
    observation_distribution=tfp.distributions.Normal(
        loc=[0.0, 5.0], scale=[1.0, 1.0]  # Means for states 0 and 1
    ),  # Standard deviations for states 0 and 1
    num_steps=100,  # Sequence length
)

# Set font for English text
plt.rcParams["font.sans-serif"] = ["Arial"]  # Font that supports English
plt.rcParams["axes.unicode_minus"] = True  # Correctly display minus sign

# Sample from the model - returns only observations
observations = hmm.sample()
print(f"Observation sequence shape: {observations.shape}")

# Forward algorithm to compute marginal likelihood
log_prob = hmm.log_prob(observations)
print(f"Sequence log likelihood: {log_prob.numpy()}")

# Viterbi algorithm to find most likely state sequence
posterior_mode = hmm.posterior_mode(observations)
print(f"Most likely state sequence shape: {posterior_mode.shape}")

# Compute posterior marginal probabilities
posterior_marginal = hmm.posterior_marginals(observations)
# Extract probabilities from Categorical distribution
posterior_probs = posterior_marginal.probs_parameter()
print(f"Posterior marginal probabilities shape: {posterior_probs.shape}")

# Visualize results
plt.figure(figsize=(15, 10))

# Plot observations
plt.subplot(3, 1, 1)
plt.plot(observations.numpy())
plt.title("Observation Sequence")
plt.ylabel("Observation Value")

# Plot most likely state sequence
plt.subplot(3, 1, 2)
plt.step(range(len(posterior_mode)), posterior_mode.numpy())
plt.title("Most Likely State Sequence (Viterbi)")
plt.ylabel("State")
plt.yticks([0, 1])

# Plot posterior probability of state 1
plt.subplot(3, 1, 3)
plt.plot(posterior_probs.numpy()[:, 1])
plt.title("Posterior Probability of State 1")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("hmm_results.png")
plt.show()

# Evaluate model performance
# Calculate proportion of correctly classified states (assuming posterior_mode is close to true states)
# In real applications, you would need true labels to compute this metric
predicted_means = tf.gather([0.0, 5.0], posterior_mode)  # Means for states 0 and 1

# Calculate mean squared error between predicted means and observations
mse = tf.reduce_mean(tf.square(predicted_means - observations))
print(f"MSE between predicted means and observations: {mse.numpy()}")

# Calculate number of state transitions
state_changes = tf.reduce_sum(
    tf.cast(tf.not_equal(posterior_mode[:-1], posterior_mode[1:]), tf.int32)
)
print(f"Number of state transitions: {state_changes.numpy()}")
print(
    f"Average probability of state transition per time step: {state_changes.numpy() / (len(posterior_mode) - 1)}"
)
