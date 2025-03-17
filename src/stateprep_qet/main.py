import numpy as np
from classiq import *
from stateprep_qet.utils import (
    amplification_phi,
    amplification_round,
    find_angle,
    h,
    h_hat,
    h_scale,
    verify,
)
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin

NUM_QUBITS = 5  # resolution of input x
F = lambda x: np.tanh(x)  # TODO: Gaussian
MIN = 0  # min x TODO: Gaussian
MAX = 1  # max x TODO: Gaussian

H = h(f=F, min=MIN, max=MAX)
POLY_FUNC = h_hat(h=H, h_max=h_scale(H))
POLY_DEGREE = 33
POLY_MAX_SCALE = 1  # TODO: Gaussian


@qfunc
def u_sin(
    x: Output[QNum],
    a1: Output[QNum],
):
    a1 *= sin(x / (2**NUM_QUBITS))  # Amplitude encoding sin(x) to |1>
    X(a1)  # sin(x) to |0>


@qfunc
def u_f(
    x: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    a3: Output[QNum],  # NOTE: a3 is unnecessary if f^{\tilde} has definite parity
):
    """u_{f^{\tilde}} circuit for state preparation using QET (more generally, QSVT)

    Args:
        x (Output[QNum]): _description_
        a1 (Output[QNum]): _description_
        a2 (Output[QNum]): _description_
        a3 (Output[QNum]): _description_
    """

    # Get phase angles for QSVT
    phiset, red_phiset, parity = find_angle(POLY_FUNC, POLY_DEGREE, POLY_MAX_SCALE)
    phase_angles = phiset

    raise NotImplementedError("TODO: Construct u_f using u_sin and phase_angles")


@qfunc
def u_amp(
    x: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    a3: Output[QNum],
    a4: Output[QNum],
):
    """Amplification circuit for state preparation using QSVT

    Args:
        x (Output[QNum]): _description_
        a1 (Output[QNum]): _description_
        a2 (Output[QNum]): _description_
        a3 (Output[QNum]): _description_
        a4 (Output[QNum]): _description_

    Raises:
        NotImplementedError: _description_
    """

    # TODO: calculate phi, round
    phi = amplification_phi()
    round = amplification_round()

    # TODO: we may use classiq's "exact amplitude amplificaiton" API instead of implementing it manually
    # link: https://docs.classiq.io/latest/qmod-reference/api-reference/functions/open_library/amplitude_amplification/?h=amplit#classiq.open_library.functions.amplitude_amplification.exact_amplitude_amplification

    raise NotImplementedError


@qfunc
def main(
    x: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    a3: Output[QNum],
    a4: Output[QNum],
    qsvt_aux: Output[QBit],
):
    allocate(NUM_QUBITS, x)
    allocate(1, a1)
    allocate(1, a2)
    allocate(1, a3)
    allocate(1, a4)
    allocate(1, qsvt_aux)

    u_amp(x, a1, a2, a3, a4, qsvt_aux)


def execute_model():
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
    result = execute(qprog).result_value()
    # show(qprog)

    verify(result)


if __name__ == "__main__":
    execute_model()
