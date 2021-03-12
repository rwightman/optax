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
"""Tests for optax._src.loss."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import numpy as np

from optax._src import loss


class L2LossTest(parameterized.TestCase):

  def setUp(self):
    super(L2LossTest, self).setUp()
    self.ys = jnp.array([-2., -1., 0.5, 1.])
    self.ts = jnp.array([-1.5, 0., -1, 1.])
    # compute expected outputs in numpy.
    self.exp = 0.5 * (self.ts - self.ys) ** 2

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(loss.l2_loss)(self.ys[0], self.ts[0]), self.exp[0])

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(loss.l2_loss)(self.ys, self.ts), self.exp)


class HuberLossTest(parameterized.TestCase):

  def setUp(self):
    super(HuberLossTest, self).setUp()
    self.ys = np.array([-2.0, 0.5, 0., 0.5, 2.0, 4.0, 132.])
    self.ts = np.array([0.0, -0.5, 0., 1., 1.0, 2.0, 0.3])
    # computed expected outputs manually.
    self.exp = np.array([1.5, 0.5, 0., 0.125, 0.5, 1.5, 131.2])

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(loss.huber_loss)(self.ys[0], self.ts[0], delta=1.0),
        self.exp[0])

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(loss.huber_loss)(self.ys, self.ts, delta=1.0),
        self.exp)


class SmoothLabelsTest(parameterized.TestCase):

  def setUp(self):
    super(SmoothLabelsTest, self).setUp()
    self.ts = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
    # compute expected outputs in numpy.
    self.exp_alpha_zero = self.ts
    self.exp_alpha_zero_point_one = 0.9 * self.ts + 0.1 / self.ts.shape[-1]
    self.exp_alpha_one = jnp.ones_like(self.ts) / self.ts.shape[-1]

  @chex.all_variants()
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts[0], 0.),
        self.exp_alpha_zero[0], atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts[0], 0.1),
        self.exp_alpha_zero_point_one[0], atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts[0], 1.),
        self.exp_alpha_one[0], atol=1e-4)

  @chex.all_variants()
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts, 0.),
        self.exp_alpha_zero, atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts, 0.1),
        self.exp_alpha_zero_point_one, atol=1e-4)
    np.testing.assert_allclose(
        self.variant(loss.smooth_labels)(self.ts, 1.),
        self.exp_alpha_one, atol=1e-4)


class SoftmaxCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super(SoftmaxCrossEntropyTest, self).setUp()
    self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
    self.ts = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
    # taken expected outputs from rlax.
    self.exp = np.array([9.00013, 3.0696733], dtype=np.float32)

  @chex.all_variants()
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.softmax_cross_entropy)(self.ys[0], self.ts[0]),
        self.exp[0], atol=1e-4)

  @chex.all_variants()
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.softmax_cross_entropy)(self.ys, self.ts),
        self.exp, atol=1e-4)


class CosineDistanceTest(parameterized.TestCase):

  def setUp(self):
    super(CosineDistanceTest, self).setUp()
    self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
    self.ts = np.array([[0., 1.2, 0.2], [1., -0.3, 0.]], dtype=np.float32)
    # computed expected output from `scipy`.
    self.exp = np.array([0.9358251989, 1.0464068465], dtype=np.float32)

  @chex.all_variants()
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.cosine_distance)(self.ys[0], self.ts[0]),
        self.exp[0], atol=1e-4)

  @chex.all_variants()
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(loss.cosine_distance)(self.ys, self.ts),
        self.exp, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
