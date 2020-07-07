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
