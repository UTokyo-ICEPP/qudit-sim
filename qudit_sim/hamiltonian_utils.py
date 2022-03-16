from typing import Union, Optional, Tuple, Callable
import re
import numpy as np

class ScaledExpression:
    """String expression with a real overall multiplier.
    """
    
    InitArg = Union['ScaledExpression', tuple, str, float]
    
    def __init__(
        self,
        arg1: InitArg,
        arg2: Optional[str] = None,
        epsilon: float = 1.e-6
    ) -> None:
        """
        ScaledExpression can be initialized by any of the following argument combinations:
        - float and str: multiplier and expression
        - tuple(float, str): multiplier and expression
        - str: expression only (multiplier set to 1)
        - float: multiplier only (expression set to None, representing unity)
        - ScaledExpression: copy construction

        When the string expression evaluates to a numerical constant, the result of `eval()` is multiplied
        to the multiplier and the internal expression is set to None.
        
        Args:
            arg1, arg2: Combinations above
            epsilon: `is_zero()` evaluates to True when `abs(scale) < epsilon`.
        """
        self._expression = None
        
        if isinstance(arg1, ScaledExpression):
            self.scale = arg1.scale
            self._expression = arg1._expression
            self.epsilon = arg1.epsilon
            return
        
        if arg2 is None and isinstance(arg1, tuple):
            arg1, arg2 = arg1
        
        if arg2 is None:
            if isinstance(arg1, str):
                try:
                    self.scale = eval(arg1)
                except NameError:
                    self.scale = 1.
                    self._expression = arg1
            else:
                self.scale = arg1
        else:
            self.scale = arg1
            try:
                self.scale *= eval(arg2)
            except (TypeError, NameError):
                self._expression = arg2

        self.epsilon = epsilon
        
    @property
    def expression(self) -> str:
        """Sanitize the expression part by enclosing it with parentheses if necessary.
        """
        if self._expression is None:
            return None
        elif (re.match(r'[a-z]*\(.+\)$', self._expression)
            or re.match(r'[^+-]+$', self._expression)):
            return self._expression
        else:
            return f'({self._expression})'
        
    def __str__(self) -> str:
        if self.is_zero():
            return '0'
        elif self._expression is None:
            return f'{self.scale}'
        elif self.scale == 1.:
            return self.expression
        elif self.scale == -1.:
            return f'-{self.expression}'
        else:
            return f'{self.scale}*{self.expression}'
        
    def __repr__(self) -> str:
        return f'ScaledExpression({self.scale}, {self._expression}, {self.epsilon})'
        
    def __add__(
        self,
        rhs: InitArg
    ) -> 'ScaledExpression':

        if isinstance(rhs, ScaledExpression):
            if self._expression is None and rhs._expression is None:
                return ScaledExpression(self.scale + rhs.scale)
            elif self.is_zero():
                return ScaledExpression(rhs)
            elif rhs.is_zero():
                return ScaledExpression(self)
            else:
                if rhs.scale < 0.:
                    return ScaledExpression(1., f'{self}-{-rhs}')
                else:
                    return ScaledExpression(1., f'{self}+{rhs}')
        else:
            if isinstance(rhs, str):
                return self.__add__(ScaledExpression(1., rhs))
            else:
                return self.__add__(ScaledExpression(rhs))
            
    def __sub__(
        self,
        rhs: InitArg
    ) -> 'ScaledExpression':
        
        return self.__add__(-rhs)
    
    def __neg__(self) -> 'ScaledExpression':
        return ScaledExpression(-self.scale, self._expression, self.epsilon)
        
    def __mul__(
        self,
        rhs: InitArg
    ) -> 'ScaledExpression':
        
        if isinstance(rhs, ScaledExpression):
            num = self.scale * rhs.scale

            if self._expression is None:
                if rhs._expression is None:
                    return ScaledExpression(num)
                else:
                    return ScaledExpression(num, rhs.expression)
            else:
                if rhs._expression is None:
                    return ScaledExpression(num, self.expression)
                else:
                    return ScaledExpression(num, f'{self.expression}*{rhs.expression}')
        else:
            if isinstance(rhs, str):
                return self.__mul__(ScaledExpression(1., rhs))
            else:
                return self.__mul__(ScaledExpression(rhs))
            
    def __abs__(self) -> 'ScaledExpression':
        if self.is_zero():
            return ScaledExpression(0., None, self.epsilon)
        elif self._expression is None:
            return ScaledExpression(abs(self.scale), None, self.epsilon)
        else:
            return ScaledExpression(abs(self.scale), f'abs({self._expression})', self.epsilon)
        
    def scale_abs(self) -> 'ScaledExpression':
        """Return a ScaledExpression where the scale is set to the abs of the scale of self.
        """
        return ScaledExpression(abs(self.scale), self._expression, self.epsilon)
            
    def is_zero(self) -> bool:
        return abs(self.scale) < self.epsilon
    

class ComplexExpression:
    """A pair of ScaledExpressions representing a complex expression.
    """
    
    def __init__(
        self,
        real: ScaledExpression.InitArg,
        imag: ScaledExpression.InitArg
    ) -> None:
        self.real = ScaledExpression(real)
        self.imag = ScaledExpression(imag)
            
    def __str__(self) -> str:
        return f'{self.real}+{self.imag}j'
    
    def __repr__(self) -> str:
        return f'ComplexExpression({self.real}, {self.imag})'
        
    def __mul__(
        self,
        rhs: Union['ComplexExpression', complex, ScaledExpression.InitArg]
    ) -> 'ComplexExpression':
        
        if isinstance(rhs, (ComplexExpression, complex)):
            real = self.real * rhs.real - self.imag * rhs.imag
            imag = self.real * rhs.imag + self.imag * rhs.real
            return ComplexExpression(real, imag)
        else:
            return ComplexExpression(self.real * rhs, self.imag * rhs)
        
    def __add__(
        self,
        rhs: Union['ComplexExpression', complex, ScaledExpression.InitArg]
    ) -> 'ComplexExpression':
        
        if isinstance(rhs, (ComplexExpression, complex)):
            real = self.real + rhs.real
            imag = self.imag + rhs.imag
            return ComplexExpression(real, imag)
        else:
            return ComplexExpression(self.real + rhs, self.imag)
        
    def __getitem__(
        self,
        key: int
    ) -> ScaledExpression:
        
        if key == 0:
            return self.real
        elif key == 1:
            return self.imag
        else:
            raise IndexError(f'Index {key} out of range')
            
    def __abs__(self) -> ScaledExpression:
        if self.real.is_zero():
            return abs(self.imag)
        elif self.imag.is_zero():
            return abs(self.real)
        elif self.real.expression == self.imag.expression:
            scale = np.sqrt(np.square(self.real.scale) + np.square(self.imag.scale))
            return ScaledExpression(scale, f'abs({self.real._expression})')
        else:
            r2 = self.real * self.real
            i2 = self.imag * self.imag
            m2 = r2 + i2
            return ScaledExpression(f'np.sqrt({m2})')
            
    def is_zero(self) -> bool:
        return self.real.is_zero() and self.imag.is_zero()
    
    def angle(self) -> float:
        """Return the static complex phase.
        
        Raises a `ValueError` if the complex phase is not static (i.e. `real` and `imag` have different
        time dependencies).
        """
        if self.imag.is_zero():
            return np.arctan2(0., self.real.scale)
        elif self.real.is_zero():
            return np.arctan2(self.imag.scale, 0.)
        elif self.real.expression == self.imag.expression:
            return np.arctan2(self.imag.scale, self.real.scale)
        else:
            raise ValueError('Angle cannot be defined for non-static phase ComplexExpression')
            
    def polar(self) -> Tuple[ScaledExpression, float]:
        """Return the polar decomposition.
        
        Raises a `ValueError` if the complex phase is not static (i.e. `real` and `imag` have different
        time dependencies).
        """
        phase = self.angle()
        
        if self.imag.is_zero():
            radial = self.real.scale_abs()
        elif self.real.is_zero():
            radial = self.imag.scale_abs()
        elif self.real.expression == self.imag.expression:
            scale = np.sqrt(np.square(self.real.scale) + np.square(self.imag.scale))
            radial = ScaledExpression(scale, self.real._expression, self.real.epsilon)

        return radial, phase
    
    
def func_prod(f1, f2):
    if f1 and f2:
        return lambda t, args: f1(t, args) * f2(t, args)
    else:
        return 0.

def func_sum(f1, f2):
    if f1 and f2:
        return lambda t, args: f1(t, args) + f2(t, args)
    elif f1:
        return f1
    elif f2:
        return f2
    else:
        return 0.

def func_diff(f1, f2):
    if f1 and f2:
        return lambda t, args: f1(t, args) - f2(t, args)
    elif f1:
        return f1
    elif f2:
        return lambda t, args: -f2(t, args)
    else:
        return 0.

def func_scale(f, s):
    if isinstance(s, np.ndarray):
        s_nonzero = s.any()
    else:
        s_nonzero = bool(s)
        
    if f and s_nonzero:
        return lambda t, args: s * f(t, args)
    else:
        return 0.
    
class ComplexFunction:
    """A pair of callables (signature (t, args)->float) representing a complex-valued function.
    """
    def __init__(
        self,
        real: Union[Callable, float],
        imag: Union[Callable, float]
    ) -> None:
        """When `real` and/or `imag` are float, the value must be 0.
        """
        self.real = real
        self.imag = imag
            
    def __repr__(self) -> str:
        return f'ComplexFunction({self.real}, {self.imag})'
        
    def __mul__(
        self,
        rhs: Union['ComplexFunction', complex, float, np.ndarray]
    ) -> 'ComplexFunction':
        
        if isinstance(rhs, ComplexFunction):
            rr = func_prod(self.real, rhs.real)
            ii = func_prod(self.imag, rhs.imag)
            ri = func_prod(self.real, rhs.imag)
            ir = func_prod(self.imag, rhs.real)
            real = func_diff(rr, ii)
            imag = func_sum(ri, ir)

        elif hasattr(rhs, 'real') and hasattr(rhs, 'imag'):
            rr = func_scale(self.real, rhs.real)
            ii = func_scale(self.imag, rhs.imag)
            ri = func_scale(self.real, rhs.imag)
            ir = func_scale(self.imag, rhs.real)
            real = func_diff(rr, ii)
            imag = func_sum(ri, ir)
        
        else:
            real = func_scale(self.real, rhs)
            imag = func_scale(self.imag, rhs)
        
        return ComplexFunction(real, imag)
        
    def __getitem__(
        self,
        key: int
    ) -> Union[Callable, float]:
        
        if key == 0:
            return self.real
        elif key == 1:
            return self.imag
        else:
            raise IndexError(f'Index {key} out of range')
