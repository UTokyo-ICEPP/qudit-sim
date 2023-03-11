from typing import Callable, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from .config import config

ArrayType = Union[np.ndarray, jnp.ndarray]
# Type for the callable time-dependent Hamiltonian coefficient
TimeType = Union[float, ArrayType]
ArgsType = Union[Dict[str, Any], Tuple[Any, ...]]
ReturnType = Union[complex, ArrayType]

@dataclass(frozen=True)
class Parameter:
    name: str

    def __mul__(self, other: Union[float, complex]):
        return ParameterFn(lambda args: other * args[0], (self.name,))

    def __rmul__(self, other: Union[float, complex]):
        return self.__mul__(other)

@dataclass
class ParameterFn:
    fn: Callable[[Tuple[Any, ...]], ReturnType]
    parameters: Tuple[str, ...]

    def _dict_arg_wrapper(self, fn):
        def wrapped_fn(args):
            args = tuple(args[key] for key in self.parameters)
            return fn(args)

        return wrapped_fn

    def __post_init__(self):
        if config.pulse_sim_solver == 'qutip':
            self.fn = self._dict_arg_wrapper(self.fn)

    def __call__(self, args: ArgsType) -> ReturnType:
        return self.fn(args)

    def __neg__(self):
        return self._unary_op(config.npmod.negative)

    @property
    def real(self):
        return self._unary_op(config.npmod.real)

    @property
    def imag(self):
        return self._unary_op(config.npmod.imag)

    def _unary_op(self, op):
        return TimeFunction(lambda t, args: op(self.fn(t, args)), self.parameters)

    def to_timefn(self):
        return TimeFunction(lambda t, args: self.__call__(args), self.parameters)

@dataclass
class TimeFunction:
    fn: Callable[[TimeType, Tuple[Any, ...]], ReturnType]
    parameters: Tuple[str, ...] = ()

    def _dict_arg_wrapper(self, fn):
        def wrapped_fn(t, args):
            args = tuple(args[key] for key in self.parameters)
            return fn(t, args)

        return wrapped_fn

    def __call__(self, t: TimeType, args: ArgsType) -> ReturnType:
        return self.fn(t, args)

    def __add__(self, other: TimeFunction):
        return self._binary_op(other, config.npmod.add)

    def __sub__(self, other: TimeFunction):
        return self._binary_op(other, config.npmod.subtract)

    def __mul__(self, other: Union[TimeFunction, float, complex, Parameter]):
        if isinstance(other, TimeFunction):
            return self._binary_op(other, config.npmod.multiply)

        elif isinstance(other, (float, complex)):
            return TimeFunction(lambda t, args: other * self.fn(t, args),
                                self.parameters)

        elif isinstance(other, Parameter):
            return TimeFunction(lambda t, args: args[0] * self.fn(t, args[1:]),
                                (scale.name,) + self.parameters)

        else:
            raise TypeError(f'Cannot multiply TimeFunction and {type(other)}')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __abs__(self):
        return self._unary_op(config.npmod.abs)

    def conjugate(self):
        return self._unary_op(config.npmod.conjugate)

    @property
    def real(self):
        return self._unary_op(config.npmod.real)

    @property
    def imag(self):
        return self._unary_op(config.npmod.imag)

    def _unary_op(self, op):
        return TimeFunction(lambda t, args: op(self.fn(t, args)), self.parameters)

    def _binary_op(self, other, op):
        if not isinstance(op, TimeFunction):
            raise TypeError(f'Operation {op} not defined between TimeFunction and {type(other)}')

        nparam1 = len(self.parameters)
        return TimeFunction(lambda t, args: op(self.fn(t, args[:nparam1]), other.fn(t, args[nparam1:])),
                            self.parameters + other.parameters)
