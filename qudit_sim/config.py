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
        self.pulse_sim_solver = 'qutip'

        self._local = threading.local()
        self._local.jax_devices = list(range(jax.local_device_count()))

        # Check 64-bit float support
        if jnp.array([0.], dtype=jnp.float64).dtype is not jnp.dtype('float64'):
            warnings.warn('The current device backend does not support 64-bit arithmetic, '
                          'or JAX was already configured in 32 bits. It is advised not to '
                          'set config.pulse_sim_solver to "jax".')

    @property
    def jax_devices(self):
        if self._local.jax_devices is None:
            raise RuntimeError('jax_devices is nullified. Are you perhaps trying to use jax in a multiprocessing call?')

        return self._local.jax_devices

    @jax_devices.setter
    def jax_devices(self, value):
        self._local.jax_devices = value

    @property
    def npmod(self):
        if self.pulse_sim_solver == 'qutip':
            return np
        elif self.pulse_sim_solver == 'jax':
            return jnp
        else:
            raise ValueError(f'Invalid value for config.pulse_sim_solver: {self.pulse_sim_solver}')

config = Config()
