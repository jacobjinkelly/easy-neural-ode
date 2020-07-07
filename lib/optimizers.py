# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: file has been modified.

"""
Utilities for optimization.
Modified from https://github.com/google/jax/blob/master/jax/experimental/optimizers.py
"""
import jax.numpy as jnp


def exponential_decay(step_size, decay_steps, decay_rate, lowest=-jnp.inf):
    """
    Exponentially decay the learning rate until <lowest> is achieved.
    """
    def schedule(i):
        """
        Return the learning rate as a function of the training iteration.
        """
        return jnp.maximum(step_size * decay_rate ** (i / decay_steps), lowest)
    return schedule
