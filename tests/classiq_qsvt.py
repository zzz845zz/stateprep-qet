# Reference: https://docs.classiq.io/latest/qmod-reference/library-reference/open-library-functions/qsvt/qsvt/
#
# NOTE: This code uses the latest version of pyqsp.

from typing import Dict, Tuple

from httplib2 import Authentication
import numpy as np
from numpy.polynomial import Polynomial
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import classiq
from classiq import *
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
import matplotlib.pyplot as plt
from pyqsp.poly import polynomial_generators, PolyTaylorSeries
from stateprep_qet.utils import find_angle

NUM_QUBITS = 2
DEGREE = 3
POLY = lambda x: 0.57226496 * x - 0.5134161 * (x**3) - 1.05884886 * (x**5)
# 0.0 + 0.57226496·x + 0.0·x² - 0.5134161·x³ + 0.0·x⁴ - 1.05884886·x⁵


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def u_sqrt(a: QNum, ref: QNum, res: QBit) -> None:
    hadamard_transform(ref)
    res ^= a <= ref


@qfunc
def qsvt_sqrt_polynomial(
    qsvt_phases: CArray[CReal], state: QNum, ref: QNum, ind: QBit, qsvt_aux: QBit
) -> None:
    full_reg = QArray[QBit]("full_reg")
    print("phases:", qsvt_phases)
    bind([ind, ref, state], full_reg)

    qsvt(
        qsvt_phases,
        lambda reg_, aux_: projector_cnot(reg_[0 : NUM_QUBITS + 1], aux_),  # ind+ref
        lambda reg_, aux_: projector_cnot(reg_[0], aux_),  # ind
        lambda reg_: u_sqrt(
            reg_[1 + NUM_QUBITS : reg_.len], reg_[1 : 1 + NUM_QUBITS], reg_[0]
        ),
        full_reg,
        qsvt_aux,
    )
    bind(full_reg, [ind, ref, state])


def adjust_qsvt_conventions(phases: np.ndarray) -> np.ndarray:
    phases = np.array(phases)
    phases = phases - np.pi / 2
    phases[0] = phases[0] + np.pi / 4
    phases[-1] = phases[-1] + np.pi / 2 + (2 * DEGREE - 1) * np.pi / 4

    # verify conventions. minus is due to exp(-i*phi*z) in qsvt in comparison to qsp
    return -2 * phases


def compute_qsvt_phases(poly):
    phiset, red_phiset, parity = QuantumSignalProcessingPhases(
        poly, method="sym_qsp", chebyshev_basis=True
    )
    return phiset, red_phiset, parity


def parse_qsvt_results(result) -> Tuple[np.ndarray, np.ndarray]:
    parsed_state_vector = result.parsed_state_vector
    d: Dict = {x: [] for x in range(2**NUM_QUBITS)}

    for parsed_state in parsed_state_vector:
        if (
            parsed_state["qsvt_aux"] == 0
            and parsed_state["ind"] == 0
            and np.linalg.norm(parsed_state.amplitude) > 1e-15
            and (DEGREE % 2 == 1 or parsed_state["ref"] == 0)
        ):
            d[parsed_state["state"]].append(parsed_state.amplitude)
            print(parsed_state)

    d = {k: np.linalg.norm(v) for k, v in d.items()}
    values = [d[i] for i in range(len(d))]
    x = np.sqrt(np.linspace(0, 1 - 1 / (2**NUM_QUBITS), 2**NUM_QUBITS))
    measured_poly_values = np.sqrt(2**NUM_QUBITS) * np.array(values)
    target_poly_values = np.abs(POLY(x))

    plt.scatter(x, measured_poly_values, label="measured", c="g")
    plt.plot(x, target_poly_values, label="target")
    plt.xlabel(r"$\sqrt{x}$")
    plt.ylabel(r"$P(\sqrt{x})$")
    plt.legend()

    return measured_poly_values, target_poly_values


@qfunc
def main(
    state: Output[QNum],
    ref: Output[QNum],
    ind: Output[QBit],
    qsvt_aux: Output[QBit],
) -> None:
    allocate(NUM_QUBITS, state)
    allocate(NUM_QUBITS, ref)
    allocate(1, ind)
    allocate(1, qsvt_aux)

    hadamard_transform(state)
    qsvt_sqrt_polynomial(QSVT_PHASES, state, ref, ind, qsvt_aux)


if __name__ == "__main__":
    classiq.authenticate()  # Uncoment to authenticate. For the first (local) run only
    phiset, red_phiset, parity = find_angle(POLY, DEGREE, 0.9)
    # QSVT_PHASES = red_phiset
    QSVT_PHASES = phiset

    print(phiset)

    qmod = create_model(
        main,
        constraints=Constraints(max_width=100),
        execution_preferences=ExecutionPreferences(
            num_shots=1,
            backend_preferences=ClassiqBackendPreferences(
                backend_name="simulator_statevector"
            ),
        ),
        out_file="qsvt",
    )

    qprog = synthesize(qmod)
    show(qprog)

    result = execute(qprog).result_value()
    measured, target = parse_qsvt_results(result)

    print(measured)
    print(target)
    # assert np.allclose(measured, target, atol=0.02)
