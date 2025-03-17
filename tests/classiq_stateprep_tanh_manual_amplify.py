from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from classiq import *
from stateprep_qet.utils import (
    amp_to_prob,
    amplification_phi,
    amplification_round,
    fidelity,
    find_angle,
    h,
    h_hat,
    h_scale,
    normalize,
)
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin, cos

NUM_QUBITS = 5  # U is applied to these qubits
POLY_DEGREE = 33
POLY_FUNC = h_hat(h=h(f=lambda x: np.tanh(x), min=0, max=1), h_max=np.tanh(1))
POLY_MAX_SCALE = 1  # TODO: is it correct?
# POLY_MAX_SCALE = h_scale(h) # TODO


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def u_sin(x: QNum, a1: QNum) -> None:
    a1 *= sin(x / (2**NUM_QUBITS))  # Amplitude encoding sin(x) to |1>
    X(a1)  # sin(x) to |0>


@qfunc
def u_f(
    x: QNum,
    a1: QNum,
    qsvt_aux: QBit,
):
    """u_{f^{\tilde}} circuit for state preparation using QET (more generally, QSVT)

    It uses u_sin and phase_angles
    """

    # Apply QSVT
    phiset, red_phiset, parity = find_angle(POLY_FUNC, POLY_DEGREE, POLY_MAX_SCALE)
    reg = QArray[QBit]("full_reg")
    bind([a1, x], reg)
    qsvt(
        phase_seq=phiset,
        proj_cnot_1=lambda reg, qsvt_aux: projector_cnot(
            reg[0], qsvt_aux
        ),  # reg==0 representing "from state". If the state is "from state", then mark aux qubit as |1>
        proj_cnot_2=lambda reg, qsvt_aux: projector_cnot(
            reg[0], qsvt_aux
        ),  # reg==0 representing "to state". If the state is "to state", then mark aux qubit as |1>
        u=lambda reg: u_sin(
            reg[1 : reg.len], reg[0]
        ),  # reg[1:reg.len] is "x" and reg[0] is "a1"
        qvar=reg,
        aux=qsvt_aux,
    )
    bind(reg, [a1, x])


@qfunc
def u_amp(x: QNum, a1: QNum, a2: QNum, a3: QNum, a4: QNum):
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

    # TODO: calculate phi
    phi = amplification_phi()

    @qfunc
    def u_round(x: QNum, a1: QNum, a2: QNum, a3: QNum, a4: QNum):

        @qfunc
        def u_round_inner(x: QNum, a1: QNum, a2: QNum, a3: QNum, a4: QNum):
            hadamard_transform(x)
            u_f(x, a1, a2, a3)
            RY(theta=phi, target=a4)

        u_round_inner(x, a1, a2, a3, a4)
        control(ctrl=(a1 == 0 and a2 == 0 and a3 == 0), stmt_block=lambda: Z(a4))
        invert(lambda: u_round_inner(x, a1, a2, a3, a4))
        control(
            ctrl=(x == 0 and a1 == 0 and a2 == 0 and a3 == 0), stmt_block=lambda: Z(a4)
        )

    # TODO: calculate amplitication_round
    repeat(
        count=amplification_round(), iteration=lambda _round: u_round(x, a1, a2, a3, a4)
    )


@qfunc
def main(
    x: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    a3: Output[QNum],
    a4: Output[QNum],
):
    allocate(NUM_QUBITS, x)
    allocate(1, a1)
    allocate(1, a2)
    allocate(1, a3)
    allocate(1, a4)

    u_amp(x, a1, a2, a3, a4)


def parse_qsvt_results(result) -> Dict:
    # print("state_vector", result.state_vector)
    amps: Dict = {x: [] for x in range(2**NUM_QUBITS)}

    for parsed_state in result.parsed_state_vector:
        if (
            parsed_state["aux"] == 0
            and parsed_state["ind"] == 0
            and np.linalg.norm(parsed_state.amplitude) > 1e-10
        ):
            # print(parsed_state)
            # print("prob:", np.abs(parsed_state.amplitude) ** 2)
            amps[parsed_state["x"]].append(parsed_state.amplitude)

    simulated_prob = normalize([amp_to_prob(amp) for amp in amps.values()])
    assert np.allclose(np.sum(simulated_prob), 1)

    return simulated_prob


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

    raise NotImplementedError("TODO")
    simulated_amplitudes = []  # TODO
    expected_amplitudes = []  # TODO
    print("simulated amplitudes:", simulated_amplitudes)
    print("expected amplitudes:", expected_amplitudes)

    # assert the probabilities sum to 1
    assert np.allclose(np.sum(np.abs(simulated_amplitudes) ** 2), 1)
    assert np.allclose(np.sum(np.abs(expected_amplitudes) ** 2), 1)

    # assert the fidelity between the simulated and expected amplitudes is close to 1
    assert 1 - fidelity(simulated_amplitudes, expected_amplitudes) < 1e-6
    print("PASSED")


if __name__ == "__main__":
    execute_model()
