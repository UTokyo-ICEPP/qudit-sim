"""Parallel execution function wrapper."""

from typing import Callable, List, Sequence, Tuple, Optional, Union, Any
import sys
import time
import traceback
import multiprocessing
import threading
import logging

from .config import config

logger = logging.getLogger(__name__)

# For whatever reason multiprocessing.Pool hangs in jupyter without ever starting the processes,
# so I'm sticking to a more crude approach

def parallel_map(
    target: Callable,
    args: Optional[List[Any]] = None,
    kwarg_keys: Optional[Union[str, Tuple[str]]] = None,
    kwarg_values: Optional[List[Any]] = None,
    common_args: Optional[tuple] = None,
    common_kwargs: Optional[dict] = None,
    arg_position: Optional[Union[int, Sequence[int]]] = None,
    thread_based: Union[bool, str] = False,
    log_level: int = logging.WARNING
) -> list:
    """Call the target on a list of args in parallel processes.

    The returned result is equivalent to

    .. code-block:: python

        kwargs = list(dict(zip(kwarg_keys, values)) for values in kwarg_values)
        [target(*(a + common_args), **(k | common_kwargs)) for a, k in zip(args, kwargs)]

    The order of the positional arguments can optionally be specified via the `arg_position` parameter. If this is
    given, the positional arguments to the target function is

    .. code-block:: python

        positional[arg_position[i]] = args[][i]

    with `common_args` filling in the unused positions in order.

    Args:
        target: Function to execute in parallel.
        args: List of positional arguments.
        kwarg_keys: Key(s) of mapped keyword arguments.
        kwarg_values: List of values of keyword arguments.
        common_args: Positional arguments common to all invocation of the function.
        common_kwargs: Keyword arguments common to all invocation of the function.
        arg_position: Positions of each element of args in the function call.
        thread_based: Use threads instead of processes. As a debugging option, string ``serial`` can be
            passed, in which case no parallelization is performed.
        log_level: logger level.

    Returns:
        List of return values of the function.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    num_cpus = config.num_cpus
    if num_cpus <= 0:
        num_cpus = multiprocessing.cpu_count()

    assert num_cpus > 0, f'Invalid num_cpus value {num_cpus}'

    arg_list = None

    if args is not None:
        if isinstance(args[0], tuple):
            arg_list = list((a, dict()) for a in args)
        else:
            arg_list = list(((a,), dict()) for a in args)

    if kwarg_keys is not None:
        if isinstance(kwarg_keys, str):
            kwargs = list({kwarg_keys: value} for value in kwarg_values)
        else:
            kwargs = list(dict(zip(kwarg_keys, values)) for values in kwarg_values)

        if arg_list is not None:
            assert len(arg_list) == len(kwarg_values), f'Inconsistent argument lengths: {len(arg_list)} != {len(kwarg_values)}'

            arg_list = list((a, k) for (a, _), k in zip(arg_list, kwargs))

        else:
            arg_list = list((tuple(), k) for k in kwargs)

    assert arg_list is not None, 'args or kwarg_keys must be set'

    logger.info('Executing %s in %d parallel (max %d simultaneous) %s..', target.__name__, len(arg_list), num_cpus,
                'threads' if thread_based else 'processes')

    num_done = 0

    processes = []
    results = [None] * len(arg_list)

    for itask, (a, k) in enumerate(arg_list):
        num_procs = len(processes)

        _wait_procs(processes, results, num_cpus - 1)

        if len(processes) < num_procs:
            num_done += num_procs - len(processes)
            logger.info(' %d processes completed', num_done)

        if common_args is not None:
            if arg_position is None:
                a += common_args
            elif isinstance(arg_position, int):
                assert len(a) == 1, 'arg_position is an integer but the number of positional arguments is not 1'
                a = common_args[:arg_position] + a + common_args[arg_position:]
            else:
                newargs = []
                common = iter(common_args)
                for iarg in range(len(a) + len(common_args)):
                    try:
                        pos_idx = arg_position.index(iarg)
                    except ValueError:
                        newargs.append(next(common))
                    else:
                        newargs.append(a[pos_idx])

                a = tuple(newargs)

        if common_kwargs is not None:
            k.update(common_kwargs)

        proc_name = f'task{itask}'

        if thread_based:
            # In thread-based parallelization, each thread sees only one GPU
            jax_device_id = config.jax_devices[itask % len(config.jax_devices)]

            if thread_based == 'serial':
                # Serial (debug) mode
                logger.debug('Running process %d', itask)
                current_jax_devices = config.jax_devices
                config.jax_devices = [jax_device_id]

                results[itask] = target(*a, **k)

                config.jax_devices = current_jax_devices

                continue
            else:
                conn = ThreadConn()
                conn_recv = conn
                proc_config = {'jax_devices': [jax_device_id]}
                proc_args = (target, a, k, conn, proc_name, proc_config)
                process = threading.Thread(target=_process_wrapper, args=proc_args, name=proc_name)
        else:
            conn_recv, conn_send = multiprocessing.Pipe()
            # JAX (or CUDA in general?) does not seem to work with multiprocessing
            proc_config = {'jax_devices': None}
            proc_args = (target, a, k, conn_send, proc_name, proc_config)
            process = multiprocessing.Process(target=_process_wrapper, args=proc_args, name=proc_name)

        process.start()
        processes.append((process, itask, conn_recv))

        logger.debug('Process %d started', itask)

    _wait_procs(processes, results, 0)

    logger.info('All processes completed')

    logger.setLevel(original_log_level)

    return results


class ThreadConn:
    def __init__(self):
        pass

    def poll(self):
        return hasattr(self, 'value')

    def send(self, value):
        self.value = value

    def recv(self):
        return self.value

    def close(self):
        del self.value


def _wait_procs(processes, results, max_num, wait=2):
    while True:
        ip = 0
        while ip < len(processes):
            process, idx, conn = processes[ip]
            if conn.poll():
                results[idx] = conn.recv()
                conn.close()

                process.join()
                processes.pop(ip)
            else:
                ip += 1

        if len(processes) > max_num:
            logger.debug('More than %d processes are running. Waiting for %f seconds', max_num, wait)
            time.sleep(wait)
        else:
            break

def _process_wrapper(target, args, kwargs, conn, proc_name, proc_config=None):
    if proc_config:
        for key, value in proc_config.items():
            setattr(config, key, value)

    try:
        result = target(*args, **kwargs)
    except Exception:
        sys.stderr.write(f'Exception in {proc_name}:\n')
        traceback.print_exc()
        conn.send(None)
    else:
        conn.send(result)
