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
            [(t >= self.start) & (t < self.end)],
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
            self.amp = np.asarray(amp / (1. - self.offset), dtype=np.complex128)

        else:
            self.amp = np.asarray(amp, dtype=np.complex128)
            self.offset = 0.
            
    def _call(self, t, args):
        x = (t - self.center) / self.sigma
        return np.asarray(self.amp * (np.exp(-np.square(x) * 0.5) - self.offset), dtype=np.complex128)
            

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

        self.amp = np.asarray(amp, dtype=np.complex128)
        self.width = width

    def _call(self, t, args):
        t1 = self.start + self.gauss_width / 2.
        t2 = t1 + self.width
        # piecewise determines the output dtype from the first argument
        tlist = np.asarray(t, dtype=np.complex128)
        return np.piecewise(tlist,
            [t < t1, t >= t2],
            [self.gauss_left._call, self.gauss_right._call, self.amp],
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

    
class Sequence(Pulse):
    def __init__(self, pulses):
        self.pulses = sorted(pulses, key=lambda p: p.start)
        assert all((self.pulses[i].end <= self.pulses[i + 1].start) for i in range(len(pulses) - 1))
        
        self.funclist = [p._call for p in self.pulses]
        
        super().__init__(self.pulses[0].start, self.pulses[-1].end - self.pulses[0].start)
        
    def _call(self, t, args):
        condlist = [((t >= p.start) & (t < p.end)) for p in self.pulses]
        # piecewise determines the output dtype from the first argument
        tlist = np.asarray(t, dtype=np.complex128)
        return np.piecewise(tlist, condlist, self.funclist, args)
