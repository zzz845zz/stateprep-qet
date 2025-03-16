import numpy as np
from classiq import *
from stateprep_qet.utils import (
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
    a3: Output[QNum],
    qsvt_aux: Output[QBit],
):
    phiset, red_phiset, parity = find_angle(POLY_FUNC, POLY_DEGREE, POLY_MAX_SCALE)
    phase_angles = phiset

    # TODO: Construct QSVT circuit using u_sin and phase_angles

    raise NotImplementedError("TODO")


@qfunc
def u_amplification(
    x: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    a3: Output[QNum],
    a4: Output[QNum],
    qsvt_aux: Output[QBit],
):
    # Construct equal superposition of all states
    hadamard_transform(x)

    # TODO: Construct full circuit by attaching amplitude amplification technique to u_f

    raise NotImplementedError("TODO")


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

    u_amplification(x, a1, a2, a3, a4, qsvt_aux)


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
