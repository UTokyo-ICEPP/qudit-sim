"""Global configuration parameters."""
import threading
import jax

class Config:
    """Global configuration parameters.

    Some parameters are held as thread-local data to allow per-thread configuration.
    """
    def __init__(self):
        self.num_cpus = 0

        self._local = threading.local()
        self._local.jax_devices = list(range(jax.local_device_count()))

    @property
    def jax_devices(self):
        if self._local.jax_devices is None:
            raise RuntimeError('jax_devices is nullified. Are you perhaps trying to use jax in a multiprocessing call?')

        return self._local.jax_devices

    @jax_devices.setter
    def jax_devices(self, value):
        self._local.jax_devices = value

config = Config()
