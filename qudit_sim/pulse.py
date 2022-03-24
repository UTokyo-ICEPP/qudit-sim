"""Pulse shape library"""

import numpy as np

s = 1.
ns = 1.e-9

class Pulse:
    @staticmethod
    def _zero(t, args):
        return np.zeros_like(t)
    
    def __init__(self, start, duration):
        assert duration > 0., 'Pulse duration must be positive'
        self.start = start
        self.end = start + duration
        
    def __call__(self, t, args):
        zero = np.array(0., dtype=np.complex128)
        fval = np.asarray(self._call(t, args), dtype=np.complex128)
        return np.select(
            [np.less(np.less_equal(self.start, t), self.end)],
            [fval], default=zero)
        

class Gaussian(Pulse):
    def __init__(self, start, duration, amp, sigma, center=None, zero_ends=True):
        super().__init__(start, duration)
        
        assert sigma > 1. * ns, 'Gaussian sigma must be greater than 1 ns'

        self.sigma = sigma

        if center is None:
            self.center = (self.end + self.start) * 0.5
        else:
            self.center = center
            
        if zero_ends:
            assert np.abs(self.center - (self.end + self.start) * 0.5) < 1. * ns, \
                'Zero-ended Gaussian must be centered'
            
            x = (self.end - self.start) * 0.5 / self.sigma
            self.offset = np.exp(-np.square(x) * 0.5)
            self.amp = amp / (1. - self.offset)

        else:
            self.amp = amp
            self.offset = 0.
            
    def _call(self, t, args):
        x = (t - self.center) / self.sigma
        return self.amp * (np.exp(-np.square(x) * 0.5) - self.offset)
            

class GaussianSquare(Pulse):
    def __init__(self, start, duration, amp, sigma, width, zero_ends=True):
        super().__init__(start, duration)
        
        assert sigma > 1. * ns, 'Gaussian sigma must be greater than 1 ns'
        assert width < duration, 'GaussianSquare width must be less than duration'
        
        self.gauss_width = duration - width
        self.gauss_left = Gaussian(start, self.gauss_width, amp=amp, sigma=sigma,
                                   center=None, zero_ends=zero_ends)
        self.gauss_right = Gaussian(start + width, self.gauss_width, amp=amp, sigma=sigma,
                                   center=None, zero_ends=zero_ends)

        self.amp = amp

    def _call(self, t, args):
        t1 = self.start + self.gauss_width / 2.
        t2 = t1 + self.width
        return np.piecewise(t,
            [t < t1, np.less(np.less_equal(t1, t), t2), t >= t2],
            [self.gauss_left._call, self.amp, self.gauss_right._call],
            args)

    
class Drag(Gaussian):
    def __init__(self, start, duration, amp, sigma, beta, center=None, zero_ends=True):
        super().__init__(
            start,
            duration,
            amp=amp,
            sigma=sigma,
            center=center,
            zero_ends=zero_ends)
        
        self.beta = beta
        
    def _call(self, t, args):
        gauss = super()._call(t, args)
        dgauss = -(t - self.center) / np.square(self.sigma) * gauss

        return gauss + 1.j * self.beta * dgauss
