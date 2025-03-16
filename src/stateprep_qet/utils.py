# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import polynomial_generators, PolyTaylorSeries
from typing import Dict


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
        cheb_samples=2 * polydeg,
    )
    print("poly:", poly)
    print("poly.coeffs:", poly.coef)

    # Compute full phases (and reduced phases, parity) using symmetric QSP.
    (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly, method="sym_qsp", chebyshev_basis=True
    )

    # true_func = lambda x: max_scale * func(x)  # For error, include scale.
    # response.PlotQSPResponse(
    #     phiset, pcoefs=poly, target=true_func, sym_qsp=True, simul_error_plot=True
    # )

    return phiset, red_phiset, parity


def get_random_unitary(num_qubits, seed=4):
    np.random.seed(seed)
    X = np.random.rand(2**num_qubits, 2**num_qubits)
    U, s, V = np.linalg.svd(X)

    unitary = U @ V.T
    A_dim = int(unitary.shape[0] / 2)
    A = unitary[:A_dim, :A_dim]
    print("A:", A)

    # Assert unitary is indeed unitary
    assert np.allclose(
        unitary @ unitary.T, np.eye(unitary.shape[0]), rtol=1e-5, atol=1e-6
    )
    return unitary


def normalize(list):
    return list / np.sum(list)


def amp_to_prob(amplitude):
    return (np.linalg.norm(amplitude)) ** 2


def verify(unitary: np.ndarray):
    A_dim = int(unitary.shape[0] / 2)
    A = unitary[:A_dim, :A_dim]
    print("A:", A)

    # Make sure the singular values for A are smaller than 1
    assert not (np.linalg.svd(A)[1] > 1).sum()

    # Verify U is indeed unitary
    assert np.allclose(
        unitary @ unitary.T, np.eye(unitary.shape[0]), rtol=1e-5, atol=1e-6
    )

    # Calculate the condition number
    kappa = max(1 / np.linalg.svd(A)[1])
    print("kappa:", kappa)
