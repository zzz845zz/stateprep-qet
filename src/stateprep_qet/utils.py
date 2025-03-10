# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)

def find_angle(func, polydeg, max_scale):
    """
    With PolyTaylorSeries class, compute Chebyshev interpolant to degree
    'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).
    """
    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=polydeg,
        max_scale=max_scale,
        chebyshev_basis=True,
        cheb_samples=2*polydeg)

    # Compute full phases (and reduced phases, parity) using symmetric QSP.
    (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly,
        method='sym_qsp',
        chebyshev_basis=True)

    return phiset, red_phiset, parity
