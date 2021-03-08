# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Standard losses used in optimisation."""

import chex
import jax
import jax.numpy as jnp
from optax._src import utils


def l2_loss(
    predictions: chex.Array,
    targets: chex.Array,
) -> chex.Array:
  """Calculates the L2 loss of predictions wrt targets.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  References:
    [Chris Bishop, 2006](https://bit.ly/3eeP0ga)

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    the squared error loss.
  """
  chex.assert_type([predictions, targets], float)
  return 0.5 * (predictions - targets)**2


def huber_loss(
    predictions: chex.Array,
    targets: chex.Array,
    delta: float = 1.) -> chex.Array:
  """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

  If gradient descent is applied to the `huber loss`, it is equivalent to
  clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

  References:
    [Huber, 1964](www.projecteuclid.org/download/pdf_1/euclid.aoms/1177703732)

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.
    delta: the bounds for the huber loss transformation, defaults at 1.

  Returns:
    a vector of same shape of `x`.
  """
  chex.assert_type([predictions, targets], float)
  error = predictions - targets
  # 0.5 * err^2                  if |err| <= d
  # 0.5 * d^2 + d * (|err| - d)  if |err| > d
  abs_error = jnp.abs(error)
  quadratic = jnp.minimum(abs_error, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_error - quadratic
  return 0.5 * quadratic ** 2 + delta * linear


def smooth_labels(
    labels: chex.Array,
    alpha: float,
) -> jnp.ndarray:
  """Apply label smoothing.

  Label smoothing is often used in combination with a cross-entropy loss.
  Smoothed labels favour small logit gaps, and it has been shown that this can
  provide better model calibration by preventing overconfident predictions.

  References:
    [Müller et al, 2019](https://arxiv.org/pdf/1906.02629.pdf)

  Args:
    labels: one hot labels to be smoothed.
    alpha: the smoothing factor, the greedy category with be assigned
      probability `(1-alpha) + alpha / num_categories`

  Returns:
    a smoothed version of the one hot input labels.

  """
  chex.assert_type([labels], float)
  num_categories = labels.shape[-1]
  return (1.0 - alpha) * labels + alpha / num_categories


def softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array:
  """Computes the softmax cross entropy between sets of logits and labels.

  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

  Args:
    logits: unnormalized log probabilities.
    labels: a valid probability distribution (non-negative, sum to 1).

  Returns:
    the cross entropy loss.
  """
  chex.assert_equal_shape([logits, labels])
  chex.assert_type([logits, labels], float)
  return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def sigmoid_cross_entropy(logits, labels):
  """Computes sigmoid cross entropy given logits and multiple class labels.

  Each element in a the `logits` vector is used to predict the chance
  `p[i] = sigmoid(logits[i])` of the corresponding label being `active`
  (i.e. the chance of `labels[i]==1). Unlike in softmax cross-entropy, more
  than one label may be active for the same sample, making this loss convenient
  for multi-label binary classification problems.

  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

  Args:
    logits: Logit output values.
    labels: Ground truth integer labels in {0, 1}.

  Returns:
    a sigmoid cross entropy loss.
  """
  chex.assert_equal_shape([logits, labels])
  chex.assert_type([logits, labels], float)
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p


def likelihood(
    predictions: chex.Array,
    labels: chex.Array
) -> chex.Array:
  """Calculates the likelihood of predictions wrt targets.

  Args:
    predictions: a vector of arbitrary shape.
    labels: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  chex.assert_equal_shape([predictions, labels])
  chex.assert_type([predictions, labels], float)
  likelihood_vals = predictions**labels * (1. - predictions)**(1. - labels)
  # Note: 0**0 evaluates to NaN on TPUs, manually set these cases to 1.
  filter_indices = jnp.logical_or(
      jnp.logical_and(labels == 1, predictions == 1),
      jnp.logical_and(labels == 0, predictions == 0))
  return jnp.where(filter_indices, 1, likelihood_vals)


def negative_log_likelihood(
    predictions: chex.Array,
    labels: chex.Array,
    epsilon: float = 1e-07
) -> chex.Array:
  """Computes the negative log likelihood loss given predictions and labels.

  This is the loss function used in (multinomial) logistic regression
  and extensions of it such as neural networks, defined as the negative
  log-likelihood of the true labels given a probabilistic classifier's
  predictions. The log loss is only defined for two or more labels.

  For a single sample with true label `yt` in {0,1} and estimated
  probability `yp` the negative log likelihood loss is:

    -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

  Args:
    predictions: The predicted values.
    labels: The ground truth values.
    epsilon: A small increment to add to avoid taking a log of zero.

  Returns:
    the negative log_likelihood loss.
  """
  chex.assert_equal_shape([predictions, labels])
  chex.assert_type([predictions, labels], float)
  # A small increment to add to avoid taking a log of zero.
  predictions = jnp.clip(predictions, epsilon, 1. - epsilon)
  # Manually unpack the log rather than compute `-jnp.log(likelihood(...)`.
  log_likelihood = (
      labels * jnp.log(predictions) + (1. - labels) * jnp.log(1. - predictions))
  return - log_likelihood


def cosine_distance(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.,
) -> chex.Array:
  """Computes the cosine distance between targets and predictions.

  References:
    [Wikipedia, 2021](https://en.wikipedia.org/wiki/Cosine_similarity)

  Args:
    predictions: The predicted vector.
    targets: Ground truth target vector.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.

  Returns:
    cosine similarity values.
  """
  chex.assert_equal_shape([targets, predictions])
  chex.assert_type([targets, predictions], float)
  # vectorize norm fn, to treat all dimensions except the last as batch dims.
  batched_norm_fn = jnp.vectorize(
      utils.safe_norm, signature='(k)->()', excluded={1})
  # normalise the last dimension of targets and predictions.
  unit_targets = targets / jnp.expand_dims(
      batched_norm_fn(targets, epsilon), axis=-1)
  unit_predictions = predictions / jnp.expand_dims(
      batched_norm_fn(predictions, epsilon), axis=-1)
  # cosine distance = 1 - cosine similarity.
  return 1. - jnp.sum(unit_targets * unit_predictions, axis=-1)
