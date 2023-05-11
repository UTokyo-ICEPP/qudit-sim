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

import copy
from abc import ABC
from numbers import Number
from typing import Callable, Optional, Union, Tuple, Dict, Any, Sequence
from types import ModuleType

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

    def __truediv__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other,
                                     lambda a1, a2, npmod: npmod.true_divide(a1, a2), 'divide')

    def __pow__(self, other: Union['Expression', array_like]) -> 'Expression':
        return type(self)._binary_op(self, other,
                                     lambda a1, a2, npmod: npmod.power(a1, a2), 'power')

    def __radd__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return other.__add__(self)
        else:
            return self.__add__(other)

    def __rsub__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return other.__sub__(self)
        else:
            return (-self).__add__(other)

    def __rmul__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return other.__mul__(self)
        else:
            return self.__mul__(other)

    def __rtruediv__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return other.__truediv__(self)
        else:
            return type(self)._unary_op(self.__truediv__(other),
                                        lambda a, npmod: npmod.reciprocal(a), 'reciprocal')

    def __rpow__(self, other: Union['Expression', array_like]) -> 'Expression':
        if isinstance(other, Expression):
            return other.__pow__(self)
        else:
            return self.__pow__(other)

    def __abs__(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.abs(a), 'abs',
                                    value_type=float)

    def __neg__(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.negative(a), 'negative')

    def conjugate(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.conjugate(a), 'conjugate')

    @property
    def real(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.real(a), 'real',
                                    value_type=float)

    @property
    def imag(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.imag(a), 'imag',
                                    value_type=float)

    def cos(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.cos(a), 'cos')

    def sin(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.sin(a), 'sin')

    def exp(self) -> 'Expression':
        return type(self)._unary_op(self,
                                    lambda a, npmod: npmod.exp(a), 'exp')

    def copy(self) -> 'Expression':
        raise NotImplementedError('To be implemented in subclasses.')


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

    @property
    def value_type(self) -> type:
        return type(self.value)

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

    def copy(self) -> 'Constant':
        return Constant(self.value)


class Parameter(ParameterExpression):
    def __init__(self, name: str, value_type: type = complex):
        self.name = name
        self.value_type = value_type

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

    def copy(self) -> 'Parameter':
        return Parameter(self.name, value_type=self.value_type)


class ParameterFunction(ParameterExpression):
    def __init__(
        self,
        fn: Callable[[Tuple[Any, ...], ModuleType], ReturnType],
        parameters: Tuple[str, ...],
        value_type: type = complex
    ):
        self.fn = fn
        self._parameters = tuple(parameters)
        self.value_type = value_type

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

    def copy(self) -> 'ParameterFunction':
        return ParameterFunction(self.fn, self._parameters, value_type=self.value_type)


class _ParameterUnaryOp(ParameterFunction):
    def __init__(
        self,
        expr: ParameterExpression,
        op: Callable[[array_like, ModuleType], ReturnType],
        opname: Optional[str] = None,
        value_type: Optional[type] = None
    ):
        self.expr = expr
        self.op = op
        self.opname = opname or op.__name__

        super().__init__(self._fn, self.expr.parameters, value_type=(value_type or expr.value_type))

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

    def copy(self) -> '_ParameterUnaryOp':
        return _ParameterUnaryOp(self.expr.copy(), self.op, opname=self.opname)

ParameterExpression._unary_op = _ParameterUnaryOp


class _ParameterBinaryOp(ParameterFunction):
    """Binary operation between ParameterExpressions."""
    def __init__(
        self,
        lexpr: ParameterExpression,
        rexpr: Union[ParameterExpression, array_like],
        op: Callable[[array_like, array_like, ModuleType], ReturnType],
        opname: Optional[str] = None,
        value_type: Optional[type] = None
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

        super().__init__(fn, parameters, value_type=(value_type or lexpr.value_type))

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

    def copy(self) -> '_ParameterBinaryOp':
        try:
            rexpr = self.rexpr.copy()
        except (AttributeError, TypeError):
            rexpr = self.rexpr

        return _ParameterBinaryOp(self.lexpr.copy(), rexpr, self.op, opname=self.opname)

ParameterExpression._binary_op = _ParameterBinaryOp


class TimeFunction(Expression):
    def __init__(
        self,
        fn: Callable[[TimeType, Tuple[Any, ...], ModuleType], ReturnType],
        parameters: Tuple[str, ...] = (),
        tzero: float = 0.,
        value_type: type = complex
    ):
        self.fn = fn
        self.tzero = tzero
        self.value_type = value_type
        self.parameters = tuple(parameters)

    def _targ(self) -> str:
        if self.tzero == 0.:
            return 't'
        else:
            return f't - {self.tzero}'

    def __str__(self) -> str:
        return f'{self.fn.__name__}({self._targ()}, ({", ".join(self.parameters)}))'

    def __repr__(self) -> str:
        parameters = ', '.join(f'"{p}"' for p in self.parameters)
        return f'TimeFunction({self.fn.__name__}, ({parameters}), {self.tzero}, {self.value_type})'

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

    def is_nonzero(self) -> bool:
        return True

    def copy(self) -> 'TimeFunction':
        return TimeFunction(self.fn, parameters=self.parameters, tzero=self.tzero,
                            value_type=self.value_type)

    def shift(self, t: float) -> 'TimeFunction':
        shifted = self.copy()
        shifted.tzero += t

        return shifted


class _TimeFunctionUnaryOp(TimeFunction):
    def __init__(
        self,
        expr: TimeFunction,
        op: Callable[[array_like, ModuleType], ReturnType],
        opname: Optional[str] = None,
        value_type: Optional[type] = None
    ):
        self.expr = expr
        self.op = op
        self.opname = opname or op.__name__

        super().__init__(self._fn, self.expr.parameters, value_type=(value_type or expr.value_type))

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

    def copy(self) -> '_TimeFunctionUnaryOp':
        return _TimeFunctionUnaryOp(self.expr.copy(), self.op, opname=self.opname)

TimeFunction._unary_op = _TimeFunctionUnaryOp


class _TimeFunctionBinaryOp(TimeFunction):
    def __init__(
        self,
        lexpr: TimeFunction,
        rexpr: Union[TimeFunction, array_like],
        op: Callable[[array_like, array_like, ModuleType], ReturnType],
        opname: Optional[str] = None,
        value_type: Optional[type] = None
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

        super().__init__(fn, parameters, value_type=(value_type or lexpr.value_type))

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

    def copy(self) -> '_TimeFunctionBinaryOp':
        try:
            rexpr = self.rexpr.copy()
        except (AttributeError, TypeError):
            rexpr = self.rexpr

        return _TimeFunctionBinaryOp(self.lexpr.copy(), rexpr, self.op, opname=self.opname)

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
            value_type = type(value)
        else:
            fn = self._fn_ParameterExpression
            parameters = value.parameters
            value_type = value.value_type

        super().__init__(fn, parameters, value_type=value_type)

    def __str__(self) -> str:
        return f'({self._targ()})->{self.value}'

    def __repr__(self) -> str:
        return f'ConstantFunction({repr(self.value)})'

    def __add__(self, other: Union[TimeFunction, array_like]) -> TimeFunction:
        if isinstance(other, ConstantFunction):
            return ConstantFunction(self.value + other.value)
        elif isinstance(other, TimeFunction) and self.value == 0.:
            return other.copy()
        else:
            return super().__add__(other)

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

    def is_nonzero(self) -> bool:
        return self.value != 0.

    def copy(self) -> 'ConstantFunction':
        try:
            value = self.value.copy()
        except (AttributeError, TypeError):
            value = self.value

        return ConstantFunction(value)


class PiecewiseFunction(TimeFunction):
    """A time function defined by Piecewise connection of a list of TimeFunctions.

    Args:
        timelist: Ordered list of N + 1 time points, where the first element is the start time
            of the first function and the last element is the end time of the last function.
            The function evaluates to 0 for t < timelist[0] and t > timelist[-1].
        funclist: List of N TimeFunctions with domain [timelist[k], timelist[k+1]].
    """
    def __init__(
        self,
        timelist: Sequence[float],
        funclist: Sequence[TimeFunction]
    ):
        self._timelist = np.array(list(timelist) + [np.inf])
        self._funclist = list(funclist)
        self._arg_indices = np.cumsum([0] + list(len(func.parameters) for func in self._funclist))

        parameters = sum((func.parameters for func in self._funclist), ())
        super().__init__(self._fn, parameters, value_type=self._funclist[0].value_type)

    def __str__(self) -> str:
        value = '{\n'
        for ifunc, func in enumerate(self._funclist):
            value += f'  {func}'
            value += f'  ({self._timelist[ifunc]:.3e} <= t < {self._timelist[ifunc + 1]:.3e})\n'
        value += '}'
        return value

    def __repr__(self) -> str:
        return f'PiecewiseFunction({len(self._funclist)} functions, {self._timelist})'

    def _fn(
        self,
        t: TimeType,
        args: Tuple[Any, ...],
        npmod: ModuleType
    ) -> ReturnType:
        t = npmod.asarray(t, dtype=self.value_type)

        def evaluated_fn(func, func_args):
            def fn(t):
                return func.evaluate(t, func_args, npmod)

            return fn

        funclist = [0.]
        for ifun, func in enumerate(self._funclist):
            start = self._arg_indices[ifun]
            end = self._arg_indices[ifun + 1]
            funclist.append(evaluated_fn(func, args[start:end]))
        funclist.append(0.)

        ifun = npmod.searchsorted(self._timelist, t, side='right')
        dims = list(range(1, len(t.shape) + 1))
        condlist = npmod.expand_dims(npmod.arange(len(funclist)), dims) == ifun

        return npmod.piecewise(t, condlist, funclist)

    def copy(self) -> 'PiecewiseFunction':
        return PiecewiseFunction(self._timelist[:-1], map(lambda f: f.copy(), self._funclist))
