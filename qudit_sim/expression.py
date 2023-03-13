r"""
====================================================
Parameter and function (:mod:`qudit_sim.expression`)
====================================================

.. currentmodule:: qudit_sim.expression

This is yet another expression representation for symbolic-ish programming, used primarily
for expressing time-dependent Hamiltonian coefficients. Using sympy was considered but we
opted for an original lightweight implementation because the represented functions have to
be pure to be compatible with JAX odeint.
"""

from typing import Callable, Optional, Union, Tuple, Dict, Any
from numbers import Number
from abc import ABC

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from .config import config

ArrayType = Union[np.ndarray, jnp.ndarray]
array_like = Union[ArrayType, Number]
# Type for the callable time-dependent Hamiltonian coefficient
TimeType = Union[float, ArrayType]
ArgsType = Union[Dict[str, Any], Tuple[Any, ...]]
ReturnType = Union[complex, ArrayType]

class Expression(ABC):
    def __add__(self, other: Union['Expression', array_like]):
        return type(self)._binary_op(self, other, config.npmod.add)

    def __sub__(self, other: Union['Expression', array_like]):
        return type(self)._binary_op(self, other, config.npmod.subtract)

    def __mul__(self, other: Union['Expression', array_like]):
        return type(self)._binary_op(self, other, config.npmod.multiply)

    def __radd__(self, other: Union['Expression', array_like]):
        return self.__add__(other)

    def __rsub__(self, other: Union['Expression', array_like]):
        return (-self).__add__(other)

    def __rmul__(self, other: Union['Expression', array_like]):
        return self.__mul__(other)

    def __abs__(self):
        return type(self)._unary_op(self, config.npmod.abs)

    def __neg__(self):
        return type(self)._unary_op(self, config.npmod.negative)

    def conjugate(self):
        return type(self)._unary_op(self, config.npmod.conjugate)

    @property
    def real(self):
        return type(self)._unary_op(self, config.npmod.real)

    @property
    def imag(self):
        return type(self)._unary_op(self, config.npmod.imag)

    @classmethod
    def _binary_op(
        cls,
        lexpr: 'Expression',
        rexpr: Union['Expression', array_like],
        op: Callable
    ) -> 'Expression':
        raise NotImplementedError('To be implemented in subclasses.')

    @classmethod
    def _unary_op(
        cls,
        expr: 'Expression',
        op: Callable
    ) -> 'Expression':
        raise NotImplementedError('To be implemented in subclasses.')


class ParameterExpression(Expression):
    def subs(self, args: Dict[str, Any]):
        raise NotImplementedError('To be implemented in subclasses.')

    def evaluate(self, args: Tuple[Any, ...]):
        raise NotImplementedError('To be implemented in subclasses.')

    @classmethod
    def _binary_op(
        cls,
        lexpr: 'ParameterExpression',
        rexpr: Union['ParameterExpression', array_like],
        op: Callable
    ) -> 'ParameterExpression':
        if isinstance(rexpr, ParameterExpression):
            def fn(args):
                l_num_params = len(lexpr.parameters)
                return op(
                    lexpr.evaluate(args[:l_num_params]),
                    rexpr.evaluate(args[l_num_params:])
                )

            return ParameterFunction(fn, lexpr.parameters + rexpr.parameters)

        elif isinstance(rexpr, (np.ndarray, jnp.ndarray, Number)):
            def fn(args):
                return op(lexpr.evaluate(args), rexpr)

            return ParameterFunction(fn, lexpr.parameters)

        else:
            raise TypeError(f'Cannot apply {op} to {type(lexpr)} and {type(rexpr)}')

    @classmethod
    def _unary_op(
        cls,
        expr: 'ParameterExpression',
        op: Callable
    ) -> 'ParameterExpression':
        def fn(args):
            return op(expr.evaluate(args))

        return ParameterFunction(fn, expr.parameters)


class Constant(ParameterExpression):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'Constant({self.value})'

    def subs(self, args: Dict[str, Any] = {}):
        return self.value

    def evaluate(self, args: Tuple[Any, ...] = ()):
        return self.value

    @property
    def parameters(self):
        return ()


class Parameter(ParameterExpression):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f'Parameter({self.name})'

    def subs(self, args: Dict[str, Any]):
        return args[self.parameters[0]]

    def evaluate(self, args: Tuple[Any, ...]):
        return args[0]

    @property
    def parameters(self):
        return (self.name,)


class ParameterFunction(ParameterExpression):
    def __init__(
        self,
        fn: Callable[[Tuple[Any, ...]], ReturnType],
        parameters: Tuple[str, ...]
    ):
        self.parameters = parameters
        self.fn = fn

    def __str__(self):
        return f'ParameterFunction({", ".join(self.parameters)})'

    def subs(self, args: Optional[Dict[str, Any]] = None):
        args = tuple(args[key] for key in self.parameters)
        return self.fn(args)

    def evaluate(self, args: Optional[Tuple[Any, ...]] = None):
        return self.fn(args)


class TimeFunction(Expression):
    def __init__(
        self,
        fn: Callable[[TimeType, Tuple[Any, ...]], ReturnType],
        parameters: Optional[Tuple[str, ...]] = None,
        tzero: float = 0.
    ):
        self.fn = fn
        self.tzero = tzero

        if parameters is None:
            self.parameters = ()
        else:
            self.parameters = parameters

    def __str__(self):
        if self.tzero == 0.:
            targ = 't'
        else:
            targ = f't - {self.tzero}'

        return f'{self.fn.__name__}({targ}, ({", ".join(self.parameters)}))'

    def __call__(self, t: TimeType, args: ArgsType = ()) -> ReturnType:
        if isinstance(args, dict):
            args = tuple(args[key] for key in self.parameters)

        return self.fn(t - self.tzero, args)

    @classmethod
    def _binary_op(
        cls,
        lexpr: 'TimeFunction',
        rexpr: Union['TimeFunction', array_like],
        op: Callable
    ) -> 'TimeFunction':
        if isinstance(rexpr, TimeFunction):
            def fn(t, args=()):
                l_num_params = len(lexpr.parameters)
                return op(
                    lexpr.fn(t - lexpr.tzero, args[:l_num_params]),
                    rexpr.fn(t - rexpr.tzero, args[l_num_params:])
                )

            return TimeFunction(fn, lexpr.parameters + rexpr.parameters)

        elif isinstance(rexpr, ParameterExpression):
            def fn(t, args=()):
                l_num_params = len(lexpr.parameters)
                return op(
                    lexpr.fn(t - lexpr.tzero, args[:l_num_params]),
                    rexpr.evaluate(args[l_num_params:])
                )

            return TimeFunction(fn, lexpr.parameters + rexpr.parameters)

        elif isinstance(rexpr, (Number, np.ndarray, jnp.ndarray)):
            def fn(t, args=()):
                return op(
                    lexpr.fn(t - lexpr.tzero, args),
                    rexpr
                )

            return TimeFunction(fn, lexpr.parameters)

        else:
            raise TypeError(f'Cannot apply {op} to {type(lexpr)} and {type(rexpr)}')

    @classmethod
    def _unary_op(
        cls,
        expr: 'TimeFunction',
        op: Callable
    ) -> 'TimeFunction':
        def fn(t, args=()):
            return op(expr.fn(t - expr.tzero, args))

        return TimeFunction(fn, expr.parameters)
