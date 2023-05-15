"""Global configuration parameters."""
import threading
import warnings
import numpy as np
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True)
import jax
import jax.numpy as jnp

class Config:
    """Global configuration parameters.

    Some parameters are held as thread-local data to allow per-thread configuration.
    """
    def __init__(self):
        self.num_cpus = 0
        self.jax_devices = list(range(jax.local_device_count()))

        # Check 64-bit float support
        if jnp.array([0.], dtype=jnp.float64).dtype is not jnp.dtype('float64'):
            warnings.warn('The current JAX device backend does not support 64-bit arithmetic, '
                          'or JAX was already configured in 32 bits. It is advised not to '
                          'use "jax" for pulse simulation.')

config = Config()
if config.jax_devices:
    jax_config.update('jax_default_device', jax.devices()[config.jax_devices[0]])
