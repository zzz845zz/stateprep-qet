import numpy as np
from classiq import *
from stateprep_qet.qsvt_mixed_parity import my_qsvt
from stateprep_qet.utils import (
    amplification_phi,
    amplification_round,
    find_angle,
    h,
    h_hat,
)
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin

from stateprep_qet.verifier import verify_result

# Parameter of the Gaussian state preparation
NUM_QUBITS = 8  # resolution of input x (`resolution` in the iQuHACK2025)
EXP_RATE = 1  # decay rate of the Gaussian
MIN = -2  # min x
MAX = 2  # max x

# Scaled function definition following the paper
F = lambda x: np.exp(-EXP_RATE * (x**2))  # Gaussian
H_FUNC = h(f=F, min=MIN, max=MAX)
POLY_FUNC = h_hat(h=H_FUNC, h_max=F(0))

# Even and odd part of the function for the mixed parity QSVT
POLY_EVEN = lambda x: (POLY_FUNC(x) + POLY_FUNC(-x))
POLY_ODD = lambda x: (POLY_FUNC(x) - POLY_FUNC(-x))

# Parameter of the QSVT
POLY_DEGREE = 25
POLY_MAX_SCALE = 1  # TODO: 이대로 둬도 되는지?


@qfunc
def u_sin(
    x: Output[QNum],
    a1: Output[QNum],
):
    a1 *= sin(x / (2**NUM_QUBITS))  # Amplitude encoding sin(x) to |1>
    X(a1)  # sin(x) to |0>


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def u_f(
    x: Output[QNum],
    a1: Output[QNum],
    a2_qsvt: Output[QBit],
    a3_qsvt: Output[QBit],
):
    """u_{f^{\tilde}} circuit for state preparation using QET (more generally, QSVT)

    Args:
        x (Output[QNum]): _description_
        a1 (Output[QNum]): _description_
        a2_qsvt (Output[QNum]): _description_
        a3_qsvt (Output[QNum]): auxiliary qubit for mixed parity QSVT (NOTE: It is unnecessary if f^{\tilde} has definite parity)
    """

    # Find phase angles
    phiset_even = find_angle(POLY_EVEN, POLY_DEGREE + 3, POLY_MAX_SCALE)
    phiset_odd = find_angle(POLY_ODD, POLY_DEGREE + 2, POLY_MAX_SCALE)

    if len(phiset_even) - 1 == len(phiset_odd):
        phiset_odd = np.append(
            phiset_odd, [0]
        )  # 규칙적인 qsvt 회로 생성을 위해, dummy 값을 넣어서 둘의 angle 길이를 맞춰줌.
    assert len(phiset_even) == len(phiset_odd)

    # Apply mixed parity QSVT
    full_reg = QArray[QBit]("full_reg")
    bind([a1, x], full_reg)
    my_qsvt(
        phase_even_seq=phiset_even,
        phase_odd_seq=phiset_odd,
        proj_cnot_1=lambda reg, aux: projector_cnot(
            reg[0], aux
        ),  # reg==0 representing "from state". If the state is "from state", then mark aux qubit as |1>
        proj_cnot_2=lambda reg, aux: projector_cnot(
            reg[0], aux
        ),  # reg==0 representing "to state". If the state is "to state", then mark aux qubit as |1>
        u=lambda reg: u_sin(
            reg[1 : reg.len], reg[0]
        ),  # reg[1:reg.len] is "x" and reg[0] is "a"
        qvar=full_reg,
        aux=a2_qsvt,
        aux2=a3_qsvt,
    )
    bind(full_reg, [a1, x])


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

    verify_result(result)


if __name__ == "__main__":
    execute_model()
