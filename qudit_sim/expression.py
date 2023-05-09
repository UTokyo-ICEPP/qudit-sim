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
from types import ModuleType
from numbers import Number
from abc import ABC

import numpy as np
import jax
import jax.numpy as jnp

ArrayType = Union[np.ndarray, jnp.ndarray]
array_like = Union[ArrayType, Number]
# Type for the callable time-dependent Hamiltonian coefficient
TimeType = Union[float, ArrayType]
ArgsType = Union[Dict[str, Any], Tuple[Any, ...]]
ReturnType = Union[float, complex, ArrayType]

class Expression(ABC):
    def __add__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other,
                                     lambda a1, a2, npmod: npmod.add(a1, a2), 'add')

    def __sub__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other,
                                     lambda a1, a2, npmod: npmod.subtract(a1, a2), 'subtract')

    def __mul__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other,
                                     lambda a1, a2, npmod: npmod.multiply(a1, a2), 'multiply')

    def __radd__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return type(self)._binary_op(other, self,
                                         lambda a1, a2, npmod: npmod.add(a1, a2), 'add')
        else:
            return self.__add__(other)

    def __rsub__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            type(self)._binary_op(other, self,
                                  lambda a1, a2, npmod: npmod.subtract(a1, a2), 'subtract')
        else:
            return (-self).__add__(other)

    def __rmul__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            type(self)._binary_op(other, self,
                                  lambda a1, a2, npmod: npmod.multiply(a1, a2), 'multiply')
        else:
            return self.__mul__(other)

    def __abs__(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.abs(a), 'abs')

    def __neg__(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.negative(a), 'negative')

    def conjugate(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.conjugate(a), 'conjugate')

    @property
    def real(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.real(a), 'real')

    @property
    def imag(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.imag(a), 'imag')

    def cos(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.cos(a), 'cos')

    def sin(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.sin(a), 'sin')

    def exp(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.exp(a), 'exp')


class ParameterExpression(Expression):
    def subs(
        self,
        npmod: ModuleType = np,
        **kwargs
    ) -> ReturnType:
        raise NotImplementedError('To be implemented in subclasses.')

    def evaluate(
        self,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        raise NotImplementedError('To be implemented in subclasses.')

    @property
    def parameters(self) -> Tuple[str, ...]:
        raise NotImplementedError('To be implemented in subclasses.')


class Constant(ParameterExpression):
    def __init__(self, value: ReturnType):
        self.value = value

    def __str__(self) -> str:
        return f'Constant({self.value})'

    def subs(
        self,
        npmod: ModuleType = np,
        **kwargs
    ) -> ReturnType:
        return self.value

    def evaluate(
        self,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        return self.value

    @property
    def parameters(self) -> Tuple[str, ...]:
        return ()


class Parameter(ParameterExpression):
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f'Parameter({self.name})'

    def __repr__(self) -> str:
        return f'Parameter("{self.name}")'

    def subs(
        self,
        npmod: ModuleType = np,
        **kwargs
    ) -> ReturnType:
        return kwargs[self.parameters[0]]

    def evaluate(
        self,
        args: Tuple[Any, ...],
        npmod: ModuleType = np
    ) -> ReturnType:
        return args[0]

    @property
    def parameters(self) -> Tuple[str, ...]:
        return (self.name,)


class ParameterFunction(ParameterExpression):
    def __init__(
        self,
        fn: Callable[[Tuple[Any, ...], ModuleType], ReturnType],
        parameters: Tuple[str, ...]
    ):
        self._parameters = parameters
        self.fn = fn

    def __str__(self) -> str:
        return f'ParameterFunction({", ".join(self._parameters)})'

    def __repr__(self) -> str:
        parameters = ', '.join(f'"{p}"' for p in self._parameters)
        return f'ParameterFunction({fn.__name__}, ({parameters}))'

    def subs(
        self,
        npmod: ModuleType = np,
        **kwargs
    ) -> ReturnType:
        args = tuple(kwargs[key] for key in self._parameters)
        return self.evaluate(args, npmod)

    def evaluate(
        self,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np,
    ) -> ReturnType:
        return self.fn(args, npmod)

    @property
    def parameters(self) -> Tuple[str, ...]:
        return self._parameters


class _ParameterUnaryOp(ParameterFunction):
    def __init__(
        self,
        expr: ParameterExpression,
        op: Callable[[array_like, ModuleType], ReturnType],
        opname: Optional[str] = None
    ):
        self.expr = expr
        self.op = op
        self.opname = opname or op.__name__

        super().__init__(self._fn, self.expr.parameters)

    def __str__(self) -> str:
        return f'{self.opname}({self.expr})'

    def __repr__(self) -> str:
        return f'_ParameterUnaryOp({repr(self.expr)}, {self.opname})'

    def _fn(
        self,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        return self.op(self.expr.evaluate(args, npmod), npmod)

ParameterExpression._unary_op = _ParameterUnaryOp


class _ParameterBinaryOp(ParameterFunction):
    """Binary operation between ParameterExpressions."""
    def __init__(
        self,
        lexpr: ParameterExpression,
        rexpr: Union[ParameterExpression, array_like],
        op: Callable[[array_like, array_like, ModuleType], ReturnType],
        opname: Optional[str] = None
    ):
        self.lexpr = lexpr
        self.rexpr = rexpr
        self.op = op
        self.opname = opname or op.__name__

        if isinstance(rexpr, ParameterExpression):
            fn = self._fn_ParameterExpression
            parameters = self.lexpr.parameters + self.rexpr.parameters
        elif isinstance(rexpr, (np.ndarray, jnp.ndarray, Number)):
            fn = self._fn_array_like
            parameters = self.lexpr.parameters
        else:
            raise TypeError(f'Cannot apply {op} to {type(lexpr)} and {type(rexpr)}')

        super().__init__(fn, parameters)

    def __str__(self) -> str:
        return f'{self.opname}({self.lexpr}, {self.rexpr})'

    def __repr__(self) -> str:
        return f'_ParameterBinaryOp({repr(self.lexpr)}, {repr(self.rexpr)}, {self.opname})'

    def _fn_ParameterExpression(
        self,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        l_num_params = len(self.lexpr.parameters)
        return self.op(
            self.lexpr.evaluate(args[:l_num_params], npmod),
            self.rexpr.evaluate(args[l_num_params:], npmod),
            npmod
        )

    def _fn_array_like(
        self,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        return self.op(self.lexpr.evaluate(args, npmod), self.rexpr, npmod)

ParameterExpression._binary_op = _ParameterBinaryOp


class TimeFunction(Expression):
    def __init__(
        self,
        fn: Callable[[TimeType, Tuple[Any, ...], ModuleType], ReturnType],
        parameters: Optional[Tuple[str, ...]] = None,
        tzero: float = 0.
    ):
        self.fn = fn
        self.tzero = tzero

        if parameters is None:
            self.parameters = ()
        else:
            self.parameters = parameters

    def _targ(self) -> str:
        if self.tzero == 0.:
            return 't'
        else:
            return f't - {self.tzero}'

    def __str__(self) -> str:
        return f'{self.fn.__name__}({self._targ()}, ({", ".join(self.parameters)}))'

    def __repr__(self) -> str:
        parameters = ', '.join(f'"{p}"' for p in self.parameters)
        return f'TimeFunction({self.fn.__name__}, ({parameters}), {self.tzero})'

    def __call__(
        self,
        t: TimeType,
        args: ArgsType = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        if isinstance(args, dict):
            args = tuple(args[key] for key in self.parameters)

        # Calling fn(t - self.tzero) would cause the JIT-compiled function
        # to create a zero tensor and subtract it each time - better to
        # check the value here
        if self.tzero:
            t = t - self.tzero

        return self.fn(t, args, npmod)

    def evaluate(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        if self.tzero:
            t = t - self.tzero

        return self.fn(t, args, npmod)


class _TimeFunctionUnaryOp(TimeFunction):
    def __init__(
        self,
        expr: TimeFunction,
        op: Callable[[array_like, ModuleType], ReturnType],
        opname: Optional[str] = None
    ):
        self.expr = expr
        self.op = op
        self.opname = opname or op.__name__

        super().__init__(self._fn, self.expr.parameters)

    def __str__(self) -> str:
        return f'{self.opname}({self.expr})'

    def __repr__(self) -> str:
        return f'_TimeFunctionUnaryOp({repr(self.expr)}, {self.opname})'

    def _fn(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        if self.expr.tzero:
            t = t - self.expr.tzero

        return self.op(self.expr.fn(t, args, npmod), npmod)

TimeFunction._unary_op = _TimeFunctionUnaryOp


class _TimeFunctionBinaryOp(TimeFunction):
    def __init__(
        self,
        lexpr: TimeFunction,
        rexpr: Union[TimeFunction, array_like],
        op: Callable[[array_like, array_like, ModuleType], ReturnType],
        opname: Optional[str] = None
    ):
        self.lexpr = lexpr
        self.rexpr = rexpr
        self.op = op
        self.opname = opname or op.__name__

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

    def __str__(self) -> str:
        return f'{self.opname}({self.lexpr}, {self.rexpr})'

    def __repr__(self) -> str:
        return f'_TimeFunctionBinaryOp({repr(self.lexpr)}, {repr(self.rexpr)}, {self.opname})'

    def _fn_TimeFunction(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
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
            self.lexpr.fn(tl, args[:l_num_params], npmod),
            self.rexpr.fn(tr, args[l_num_params:], npmod),
            npmod
        )

    def _fn_ParameterExpression(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        l_num_params = len(self.lexpr.parameters)

        if self.lexpr.tzero:
            t = t - self.lexpr.tzero

        return self.op(
            self.lexpr.fn(t, args[:l_num_params], npmod),
            self.rexpr.evaluate(args[l_num_params:], npmod),
            npmod
        )

    def _fn_array_like(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        if self.lexpr.tzero:
            t = t - self.lexpr.tzero

        return self.op(
            self.lexpr.fn(t, args, npmod),
            self.rexpr,
            npmod
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

    def __str__(self) -> str:
        return f'({self._targ()})->{self.value}'

    def __repr__(self) -> str:
        return f'ConstantFunction({repr(self.value)})'

    def _fn_Number(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        return (t * 0.) + self.value

    def _fn_ParameterExpression(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        return (t * 0.) + self.value.evaluate(args, npmod)


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

    def __str__(self) -> str:
        value = '{\n'
        for ifunc, func in enumerate(self.funclist):
            value += f'  {func}  ({self.timelist[ifunc]} <= t < {self.timelist[ifunc + 1]})\n'
        value += '}'
        return value

    def __repr__(self) -> str:
        return f'PiecewiseFunction({len(self.funclist)} functions, {self.timelist})'

    def _fn(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        t = npmod.asarray(t)

        if npmod is jnp and len(t.shape) == 0:
            iarg_starts = np.cumsum([0] + list(len(func.parameters) for func in self.funclist))

            def make_shifted_fun(ifun):
                if ifun == 0:
                    def fn(t, args):
                        return 0.
                else:
                    func = self.funclist[ifun - 1]
                    t0 = self.timelist[ifun - 1]
                    iarg_start = iarg_starts[ifun - 1]
                    iarg_end = iarg_starts[ifun]
                    def fn(t, args):
                        return func.evaluate(t - t0, args[iarg_start:iarg_end], npmod)

                return fn

            timelist = list(self.timelist)
            timelist.append(jnp.inf)
            indices = list(range(len(self.timelist))) + [0]
            funclist = list(make_shifted_fun(ifun) for ifun in indices)

            ifun = jnp.searchsorted(jnp.array(timelist), t, side='right')

            return jax.lax.switch(ifun, funclist, t, args)

        else:
            result = 0.
            iarg = 0

            for time, func in zip(self.timelist[:-1], self.funclist):
                nparam = len(func.parameters)

                result = npmod.where(
                    t > time,
                    func.fn(t - time, args[iarg:iarg + nparam], npmod),
                    result
                )
                iarg += nparam

            result = npmod.where(t > self.timelist[-1], 0., result)
            return result
