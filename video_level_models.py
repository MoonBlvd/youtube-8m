# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class RGBSingleStream(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   prefix="",
                   **unused_params):
    rgb_fc1 = slim.fully_connected(
      model_input,
      4096,
      activation_fn=tf.nn.relu,
      scope=prefix+"rgb_fc1"
    )

    rgb_fc2 = slim.fully_connected(
      rgb_fc1,
      4096,
      activation_fn=tf.nn.relu,
      scope=prefix+"rgb_fc2"
    )

    rgb_fc3 = slim.fully_connected(
      rgb_fc2,
      vocab_size,
      activation_fn=None,
      scope=prefix+"rgb_fc3"
    )
    output = tf.nn.softmax(rgb_fc3)
    return {"predictions":output, "inference_out":rgb_fc3}

class AudioSingleStream(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   prefix="",
                   **unused_params):
    audio_fc1 = slim.fully_connected(
      model_input,
      4096,
      activation_fn=tf.nn.relu,
      scope=prefix+"audio_fc1"
    )

    audio_fc2 = slim.fully_connected(
      audio_fc1,
      4096,
      activation_fn=tf.nn.relu,
      scope=prefix+"audio_fc2"
    )

    audio_fc3 = slim.fully_connected(
      audio_fc2,
      vocab_size,
      activation_fn=None,
      scope=prefix+"audio_fc3"
    )
    output = tf.nn.softmax(audio_fc3)
    return {"predictions":output, "inference_out":audio_fc3}

class TwoStreamLateFusion(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    rgb_input, audio_input = tf.split(model_input, [1024, 128], 1)
    rgb_output = RGBSingleStream.create_model(rgb_input, vocab_size, l2_penalty=l2_penalty, prefix="")
    audio_output = AudioSingleStream.create_model(audio_input, vocab_size, l2_penalty=l2_penalty, prefix="")
    output_sum = tf.add(rgb_output["predictions"], audio_output["predictions"])
    scalar = tf.constant(0.5)
    output = tf.scalar_mul(scalar, output_sum)
    return {"predictions": output}

class TwoStreamEarlyFusion(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   use_in_hybrid=False
                   **unused_params):
    restore_concate_fc = False
    prefix = ""
    if use_in_hybrid:
      restore_concate_fc = True
      prefix = "hybrid_"
    rgb_input, audio_input = tf.split(model_input, [1024, 128], 1)
    rgb_output = RGBSingleStream.create_model(rgb_input, vocab_size, l2_penalty=l2_penalty, prefix=prefix)
    audio_output = AudioSingleStream.create_model(audio_input, vocab_size, l2_penalty=l2_penalty, prefix=prefix)
    concate = tf.concat([rgb_output["inference_output"], audio_output["inference_output"]], 1)
    concate_fc = tf.slim.fully_connected(
      concate,
      vocab_size,
      activation=None,
      scope=prefix+"concate_fc",
      restore=restore_concate_fc
    )
    output = tf.nn.softmax(concate_fc)
    return {"predictions": output}

class TwoStreamHybridFusion(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    rgb_input, audio_input = tf.split(model_input, [1024, 128], 1)
    rgb_single_output = RGBSingleStream.create_model(rgb_input, vocab_size, l2_penalty=l2_penalty)
    audio_single_output = AudioSingleStream.create_model(audio_input, vocab_size, l2_penalty=l2_penalty)
    early_fusion_output = TwoStreamEarlyFusion(model_input, vocab_size, l2_penalty=l2_penalty, use_in_hybrid=True)
    output_sum = tf.accumulate_n([rgb_single_output["predictions"], audio_single_output["predictions"], early_fusion_output["predictions"]])
    scalar = tf.constant(0.333)
    output = tf.scalar_mul(scalar, output_sum)
    return {"predictions": output}


class TwoStreamDeepRelu(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    rgb_input, audio_input = tf.split(model_input, [1024, 128], 1)
    rgb_fc1 = slim.fully_connected(
        rgb_input,
        512,
        activation_fn=tf.nn.relu,
        scope="rgb_fc1")
    rgb_fc2 = slim.fully_connected(
        rgb_input,
        2048,
        activation_fn=tf.nn.relu,
        scope="rgb_fc2")
    rgb_fc3 = slim.fully_connected(
        rgb_input,
        4096,
        activation_fn=tf.nn.relu,
        scope="rgb_fc3")

    audio_fc1 = slim.fully_connected(
        audio_input,
        64,
        activation_fn=tf.nn.relu,
        scope="audio_fc1")
    audio_fc2 = slim.fully_connected(
        rgb_input,
        1024,
        activation_fn=tf.nn.relu,
        scope="audio_fc2")
    audio_fc3 = slim.fully_connected(
        rgb_input,
        2048,
        activation_fn=tf.nn.relu,
        scope="audio_fc3")

    concate_rgb_audio = tf.concat([rgb_fc3, audio_fc3], 1)
    concate_fc = slim.fully_connected(
        concate_rgb_audio,
        vocab_size,
        activation_fn=tf.nn.relu,
        scope="final_fc")
    output = tf.nn.softmax(concate_fc)
    return {"predictions": output}
