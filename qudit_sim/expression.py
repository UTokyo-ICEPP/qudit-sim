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

from typing import Callable, Optional, Union, Tuple, Dict, Any, Sequence
from numbers import Number
from abc import ABC

import numpy as np
import jax.numpy as jnp

from .config import config

ArrayType = Union[np.ndarray, jnp.ndarray]
array_like = Union[ArrayType, Number]
# Type for the callable time-dependent Hamiltonian coefficient
TimeType = Union[float, ArrayType]
ArgsType = Union[Dict[str, Any], Tuple[Any, ...]]
ReturnType = Union[float, complex, ArrayType]

class Expression(ABC):
    def __add__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other, config.npmod.add)

    def __sub__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other, config.npmod.subtract)

    def __mul__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other, config.npmod.multiply)

    def __radd__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return type(self)._binary_op(other, self, config.npmod.add)
        else:
            return self.__add__(other)

    def __rsub__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            type(self)._binary_op(other, self, config.npmod.subtract)
        else:
            return (-self).__add__(other)

    def __rmul__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            type(self)._binary_op(other, self, config.npmod.multiply)
        else:
            return self.__mul__(other)

    def __abs__(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.abs)

    def __neg__(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.negative)

    def conjugate(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.conjugate)

    @property
    def real(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.real)

    @property
    def imag(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.imag)

    def cos(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.cos)

    def sin(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.sin)

    def exp(self) -> 'Expression':
        return type(self)._unary_op(self, config.npmod.exp)


class ParameterExpression(Expression):
    def subs(self, **kwargs) -> ReturnType:
        raise NotImplementedError('To be implemented in subclasses.')

    def evaluate(self, args: Tuple[Any, ...]) -> ReturnType:
        raise NotImplementedError('To be implemented in subclasses.')

    @property
    def parameters(self) -> Tuple[str, ...]:
        raise NotImplementedError('To be implemented in subclasses.')


class Constant(ParameterExpression):
    def __init__(self, value: ReturnType):
        self.value = value

    def __str__(self) -> str:
        return f'Constant({self.value})'

    def subs(self, **kwargs) -> ReturnType:
        return self.value

    def evaluate(self, args: Tuple[Any, ...] = ()) -> ReturnType:
        return self.value

    @property
    def parameters(self) -> Tuple[str, ...]:
        return ()


class Parameter(ParameterExpression):
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f'Parameter({self.name})'

    def subs(self, **kwargs) -> ReturnType:
        return kwargs[self.parameters[0]]

    def evaluate(self, args: Tuple[Any, ...]) -> ReturnType:
        return args[0]

    @property
    def parameters(self) -> Tuple[str, ...]:
        return (self.name,)


class ParameterFunction(ParameterExpression):
    def __init__(
        self,
        fn: Callable[[Tuple[Any, ...]], ReturnType],
        parameters: Tuple[str, ...]
    ):
        self._parameters = parameters
        self.fn = fn

    def __str__(self) -> str:
        return f'ParameterFunction({", ".join(self._parameters)})'

    def subs(self, **kwargs) -> ReturnType:
        args = tuple(kwargs[key] for key in self._parameters)
        return self.evaluate(args)

    def evaluate(self, args: Optional[Tuple[Any, ...]] = None) -> ReturnType:
        return self.fn(args)

    @property
    def parameters(self) -> Tuple[str, ...]:
        return self._parameters


class _ParameterUnaryOp(ParameterFunction):
    def __init__(
        self,
        expr: ParameterExpression,
        op: Callable
    ):
        self.expr = expr
        self.op = op

        super().__init__(self._fn, self.expr.parameters)

    def _fn(self, args: Optional[Tuple[Any, ...]] = None) -> ReturnType:
        return self.op(self.expr.evaluate(args))

ParameterExpression._unary_op = _ParameterUnaryOp


class _ParameterBinaryOp(ParameterFunction):
    """Binary operation between ParameterExpressions."""
    def __init__(
        self,
        lexpr: ParameterExpression,
        rexpr: Union[ParameterExpression, array_like],
        op: Callable
    ):
        self.lexpr = lexpr
        self.rexpr = rexpr
        self.op = op

        if isinstance(rexpr, ParameterExpression):
            fn = self._fn_ParameterExpression
            parameters = self.lexpr.parameters + self.rexpr.parameters
        elif isinstance(rexpr, (np.ndarray, jnp.ndarray, Number)):
            fn = self._fn_array_like
            parameters = self.lexpr.parameters
        else:
            raise TypeError(f'Cannot apply {op} to {type(lexpr)} and {type(rexpr)}')

        super().__init__(fn, parameters)

    def _fn_ParameterExpression(self, args: Optional[Tuple[Any, ...]] = None) -> ReturnType:
        l_num_params = len(self.lexpr.parameters)
        return self.op(
            self.lexpr.evaluate(args[:l_num_params]),
            self.rexpr.evaluate(args[l_num_params:])
        )

    def _fn_array_like(self, args: Optional[Tuple[Any, ...]] = None) -> ReturnType:
        return self.op(self.lexpr.evaluate(args), self.rexpr)

ParameterExpression._binary_op = _ParameterBinaryOp


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

    def __str__(self) -> str:
        if self.tzero == 0.:
            targ = 't'
        else:
            targ = f't - {self.tzero}'

        return f'{self.fn.__name__}({targ}, ({", ".join(self.parameters)}))'

    def __call__(self, t: TimeType, args: ArgsType = ()) -> ReturnType:
        if isinstance(args, dict):
            args = tuple(args[key] for key in self.parameters)

        # Calling fn(t - self.tzero) would cause the JIT-compiled function
        # to create a zero tensor and subtract it each time - better to
        # check the value here
        if self.tzero:
            t = t - self.tzero

        return self.fn(t, args)

    def evaluate(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        if self.tzero:
            t = t - self.tzero

        return self.fn(t, args)


class _TimeFunctionUnaryOp(TimeFunction):
    def __init__(
        self,
        expr: TimeFunction,
        op: Callable
    ):
        self.expr = expr
        self.op = op

        super().__init__(self._fn, self.expr.parameters)

    def _fn(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        if self.expr.tzero:
            t = t - self.expr.tzero

        return self.op(self.expr.fn(t, args))

TimeFunction._unary_op = _TimeFunctionUnaryOp


class _TimeFunctionBinaryOp(TimeFunction):
    def __init__(
        self,
        lexpr: TimeFunction,
        rexpr: Union[TimeFunction, array_like],
        op: Callable
    ):
        self.lexpr = lexpr
        self.rexpr = rexpr
        self.op = op

        if isinstance(rexpr, TimeFunction):
            fn = self._fn_TimeFunction
            parameters = lexpr.parameters + rexpr.parameters
        elif isinstance(rexpr, ParameterExpression):
            fn = self._fn_ParameterExpression
            parameters = lexpr.parameters + rexpr.parameters
        elif isinstance(rexpr, (Number, np.ndarray, jnp.ndarray)):
            fn = self._fn_array_like
            parameters = lexpr.parameters
        else:
            raise TypeError(f'Cannot apply {op} to {type(lexpr)} and {type(rexpr)}')

        super().__init__(fn, parameters)

    def _fn_TimeFunction(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        l_num_params = len(self.lexpr.parameters)
        if self.lexpr.tzero:
            tl = t - self.lexpr.tzero
        else:
            tl = t

        if self.rexpr.tzero:
            tr = t - self.rexpr.tzero
        else:
            tr = t

        return self.op(
            self.lexpr.fn(tl, args[:l_num_params]),
            self.rexpr.fn(tr, args[l_num_params:])
        )

    def _fn_ParameterExpression(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        l_num_params = len(self.lexpr.parameters)

        if self.lexpr.tzero:
            t = t - self.lexpr.tzero

        return self.op(
            self.lexpr.fn(t, args[:l_num_params]),
            self.rexpr.evaluate(args[l_num_params:])
        )

    def _fn_array_like(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        if self.lexpr.tzero:
            t = t - self.lexpr.tzero

        return self.op(
            self.lexpr.fn(t, args),
            self.rexpr
        )

TimeFunction._binary_op = _TimeFunctionBinaryOp


class ConstantFunction(TimeFunction):
    def __init__(
        self,
        value: Union[Number, ParameterExpression]
    ):
        self.value = value
        if isinstance(value, Number):
            fn = self._fn_Number
            parameters = ()
        else:
            fn = self._fn_ParameterExpression
            parameters = value.parameters

        super().__init__(fn, parameters)

    def _fn_Number(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        return (t * 0.) + self.value

    def _fn_ParameterExpression(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        return (t * 0.) + self.value.evaluate(args)


class PiecewiseFunction(TimeFunction):
    """A time function defined by Piecewise connection of a list of TimeFunctions.

    Args:
        timelist: Ordered list of N + 1 time points, where the first element is the start time
            of the first function and the last element is the end time of the last function.
            The function evaluates to 0 for t < timelist[0] and t > timelist[-1].
        funclist: List of N TimeFunctions.
    """
    def __init__(
        self,
        timelist: Sequence[float],
        funclist: Sequence[TimeFunction]
    ):
        self.timelist = timelist
        self.funclist = funclist

        parameters = sum((func.parameters for func in funclist), ())
        super().__init__(self._fn, parameters)

    def _fn(self, t: TimeType, args: Tuple[Any, ...] = ()) -> ReturnType:
        npmod = config.npmod
        t = npmod.asarray(t)

        result = 0.
        iarg = 0
        for time, func in zip(self.timelist[:-1], self.funclist):
            nparam = len(func.parameters)

            if func.tzero:
                tfunc = t - func.tzero
            else:
                tfunc = t

            result = npmod.where(
                t > time,
                func.fn(tfunc, args[iarg:iarg + nparam]),
                result
            )
            iarg += nparam

        result = npmod.where(t > self.timelist[-1], 0., result)
        return result
