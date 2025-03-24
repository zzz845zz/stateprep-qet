# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import polynomial_generators, PolyTaylorSeries
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from typing import Dict


def find_angle(func, polydeg, max_scale, encoding="amplitude"):
    """
    With PolyTaylorSeries class, compute Chebyshev interpolant to degree
    'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).
    """

    if encoding == "amplitude":
        phiset = compute_qsvt_phases(poly=func, degree=polydeg, max_scale=max_scale)
        return phiset
    elif encoding == "imaginary":
        # Compute full phases (and reduced phases, parity) using symmetric QSP.
        poly = PolyTaylorSeries().taylor_series(
            func=func,
            degree=polydeg,
            max_scale=max_scale,
            chebyshev_basis=True,
            cheb_samples=2 * polydeg,
        )

        (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            poly, method="sym_qsp", chebyshev_basis=True
        )
        # (phiset) = angle_sequence.QuantumSignalProcessingPhases(poly, method="laurent")

        # true_func = lambda x: max_scale * func(x)  # For error, include scale.
        # response.PlotQSPResponse(
        #     phiset, pcoefs=poly, target=true_func, sym_qsp=True, simul_error_plot=True
        # )

        return phiset, red_phiset, parity
    else:
        raise ValueError("Invalid encoding type.")


def adjust_qsvt_conventions(phases: np.ndarray, degree: int) -> np.ndarray:
    phases = np.array(phases)
    phases = phases - np.pi / 2
    phases[0] = phases[0] + np.pi / 4
    phases[-1] = phases[-1] + np.pi / 2 + (2 * degree - 1) * np.pi / 4

    # verify conventions. minus is due to exp(-i*phi*z) in qsvt in comparison to qsp
    return -2 * phases


def compute_qsvt_phases(poly, degree, max_scale):
    chebyshev_poly = PolyTaylorSeries().taylor_series(
        func=poly,
        degree=degree,
        max_scale=max_scale,
    )
    phases = QuantumSignalProcessingPhases(
        chebyshev_poly, signal_operator="Wx", method="laurent", measurement="x"
    )
    return adjust_qsvt_conventions(phases, degree).tolist()


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


def h(f, min, max):
    """
    Eq. (3) in https://arxiv.org/pdf/2210.14892
    """
    return lambda y: f((max - min) * np.arcsin(y) + min)


def h_scale(h):
    """
    Maximal value of h(y) in the y interval [0, sin(1)]. (i.e, maximal value of f(x) in the x interval [a, b])
    """
    raise NotImplementedError


def h_hat(h, h_max):
    """
    Eq. (4) in https://arxiv.org/pdf/2210.14892
    """
    return lambda y: h(y) / h_max


def discretized_l2_norm(f, N, min, max):
    """
    Compute the discretized L2-norm of the function f over the interval [a, b] with N points.

    Eq. (6) in https://arxiv.org/pdf/2210.14892

    Args:
        f (function): The function to evaluate.
        N (int): The number of discretization points.
        min (float): The start of the interval.
        max (float): The end of the interval.

    Returns:
        float: The discretized L2-norm of the function.
    """
    x = np.linspace(min, max, N)
    f_values = f(x)
    l2_norm = np.sqrt((max - min) / N * np.sum(np.abs(f_values) ** 2))
    return l2_norm


def l2_norm_filling_fraction(f, N, min, max):
    """
    Compute the L2-norm filling-fraction of the function f over the interval [a, b] with N points.

    Eq. (7) in https://arxiv.org/pdf/2210.14892

    Args:
        f (function): The function to evaluate.
        N (int): The number of discretization points.
        min (float): The start of the interval.
        max (float): The end of the interval.

    Returns:
        float: The L2-norm filling-fraction of the function.
    """
    l2_norm_discretized = discretized_l2_norm(f, N, min, max)
    f_max = np.max(np.abs(f(np.linspace(min, max, N))))
    # l2_norm_continuous = np.sqrt(np.trapz(np.abs(f(np.linspace(a, b, 1000))) ** 2, np.linspace(a, b, 1000)))
    filling_fraction = l2_norm_discretized / np.sqrt((max - min) * f_max**2)
    return filling_fraction


def fidelity(state1, state2):
    """Compute the fidelity between two states.

    Args:
        state1 (np.ndarray): list of amplitudes of state 1
        state2 (np.ndarray): list of amplitudes of state 2

    Returns:
        float: fidelity between state
    """
    return np.abs(np.dot(state1.conj().T, state2)) ** 2


def amplification_phi():
    raise NotImplementedError


def amplification_round():
    raise NotImplementedError
