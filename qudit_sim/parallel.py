from typing import Callable, List, Sequence, Optional, Union, Any
import time
from multiprocessing import Process, Pipe, cpu_count
from threading import Thread
import logging

logger = logging.getLogger(__name__)

# For whatever reason multiprocessing.Pool hangs in jupyter without ever starting the processes,
# so I'm sticking to a more crude approach

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
            
def _process_wrapper(target, args, kwargs, conn):
    result = target(*args, **kwargs)
    conn.send(result)

def parallel_map(
    target: Callable,
    mapped_args: Optional[List[Any]] = None,
    mapped_kwargs: Optional[List[dict]] = None,
    common_args: Optional[tuple] = None,
    common_kwargs: Optional[dict] = None,
    arg_position: Optional[Union[int, Sequence]] = None,
    num_cpus: int = 0,
    thread_based: bool = False,
    log_level: int = logging.WARNING
) -> list:
    """Call the target on a list of args in parallel processes.
    
    The returned result is equivalent to
    ```
    [target(*(args + common_args), **(kwargs | common_kwargs)) for args, kwargs in zip(mapped_args, mapped_kwargs)]
    ```
    The order of the positional arguments can optionally be specified via the `arg_position` parameter. If this is
    given, the positional arguments to the target function is
    ```
    positional[arg_position[i]] = mapped_args[][i]
    ```
    with `common_args` filling in the unused positions in order.
    
    Args:
        target: Function to execute in parallel.
        mapped_args: List of positional arguments.
        mapped_kwargs: List of keyword arguments.
        common_args: Positional arguments common to all invocation of the function.
        common_kwargs: Keyword arguments common to all invocation of the function.
        arg_position: Positions of each element of mapped_args in the function call.
        num_cpus: Maximum number of processes to run concurrently.
        thread_based: Use threads instead of processes.
        log_level: logger level.
        
    Returns:
        List of return values of the function.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    if num_cpus <= 0:
        num_cpus = cpu_count()
            
    assert num_cpus > 0, f'Invalid num_cpus value {num_cpus}'
    
    arg_list = None

    if mapped_args is not None:
        if isinstance(mapped_args[0], tuple):
            arg_list = list((args, dict()) for args in mapped_args)
        else:
            arg_list = list(((arg,), dict()) for arg in mapped_args)
        
    if mapped_kwargs is not None:
        if arg_list is not None:
            assert len(arg_list) == len(mapped_kwargs)
            arg_list = list((args, kwargs) for (args, _), kwargs in zip(arg_list, mapped_kwargs))
        else:
            arg_list = list((tuple(), kwargs) for kwargs in mapped_kwargs)
            
    assert arg_list is not None
    
    logger.info('Starting %d parallel execution of %s', len(arg_list), target.__name__)
    
    num_done = 0

    processes = []
    results = [None] * len(arg_list)
    
    for itask, (args, kwargs) in enumerate(arg_list):
        num_procs = len(processes)
        
        _wait_procs(processes, results, num_cpus - 1)
        
        if len(processes) < num_procs:
            num_done += num_procs - len(processes)
            logger.info(' %d processes completed', num_done)
        
        if common_args is not None:
            if arg_position is None:
                args += common_args
            elif isinstance(arg_position, int):
                assert len(args) == 1
                args = common_args[:arg_position] + args + common_args[arg_position:]
            else:
                newargs = []
                common = iter(common_args)
                for iarg in range(len(args) + len(common_args)):
                    try:
                        pos_idx = arg_position.index(iarg)
                    except ValueError:
                        newargs.append(next(common))
                    else:
                        newargs.append(args[pos_idx])
                        
                args = tuple(newargs)
                
        if common_kwargs is not None:
            kwargs.update(common_kwargs)

        if thread_based:
            conn = ThreadConn()
            conn_recv = conn
            proc_args = (target, args, kwargs, conn)
            process = Thread(target=_process_wrapper, args=proc_args)
        else:
            conn_recv, conn_send = Pipe()
            proc_args = (target, args, kwargs, conn_send)
            process = Process(target=_process_wrapper, args=proc_args)

        process.start()
        processes.append((process, itask, conn_recv))
        
        logger.debug('Process %d started', itask)
        
    _wait_procs(processes, results, 0)

    logger.info('All processes completed')
    
    logger.setLevel(original_log_level)
    
    return results
