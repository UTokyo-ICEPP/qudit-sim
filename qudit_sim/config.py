"""Global configuration parameters."""
import threading

class Config:
    """Global configuration parameters.

    Some parameters are held as thread-local data to allow per-thread configuration.
    """
    def __init__(self):
        self.num_cpus = 0

        self._local = threading.local()
        self._local.jax_devices = None

    @property
    def jax_devices(self):
        if self._local.jax_devices is None:
            try:
                import jax
            except ImportError:
                self._local.jax_devices = []
            else:
                self._local.jax_devices = list(range(jax.local_device_count()))

        return self._local.jax_devices

    @jax_devices.setter
    def jax_devices(self, value):
        self._local.jax_devices = value

config = Config()
