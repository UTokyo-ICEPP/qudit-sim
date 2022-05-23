"""Global configuration parameters.

Because we want to be using independenty configuration parameters in parallel threads, we define
`config` as a thread-local data storage.
"""
import threading

config = threading.local()

config.num_cpus = 0
"""Number of threads to use in parallelization routines. If <=0, set to `multiprocessing.cpu_count()`.
For extraction methods that use GPUs, the combination of `jax_devices` and this parameter controls
how many processes will be run on each device."""

try:
    import jax
except ImportError:
    config.jax_devices = []
else:
    config.jax_devices = list(range(jax.local_device_count()))
    """List of GPU ids (integers starting at 0) to use."""
