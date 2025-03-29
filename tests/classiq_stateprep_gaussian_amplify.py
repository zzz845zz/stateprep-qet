import argparse
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from classiq import *
from stateprep_qet.utils import (
    amp_to_prob,
    find_angle,
    get_Amplitude_Gaussian_Fixed,
    h,
    h_hat,
    normalize,
)
from stateprep_qet.qsvt_mixed_parity import my_qsvt
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin


NUM_QUBITS = 8  # resolution of input x
EXP_RATE = 1  # decay rate of the Gaussian

F = lambda x: np.exp(-EXP_RATE * (x**2))  # Gaussian
MIN = -2  # min x
MAX = 2  # max x

H_FUNC = h(f=F, min=MIN, max=MAX)
POLY_FUNC = h_hat(h=H_FUNC, h_max=F(0))
POLY_DEGREE = 25
POLY_MAX_SCALE = 1

POLY_EVEN = lambda x: (POLY_FUNC(x) + POLY_FUNC(-x))
POLY_ODD = lambda x: (POLY_FUNC(x) - POLY_FUNC(-x))

AMPLITUDE = 0.5597575631451602


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def u_sin(x: QNum, a: QNum) -> None:
    a *= sin(x / (2**NUM_QUBITS))  # Amplitude encoding sin(x) to |1>
    X(a)  # sin(x) to |0>


@qfunc
def u_f(
    x: QNum,
    a1: QNum,
    a2_qsvt: QBit,
    a3_qsvt: QBit,
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
    H(a3_qsvt)


@qfunc
def state_prep(reg: QArray[QBit]):
    # reg[0:NUM_QUBITS]: x
    # reg[NUM_QUBITS]: a1
    # reg[NUM_QUBITS + 1]: a2
    # reg[NUM_QUBITS + 2]: a3
    hadamard_transform(reg[0:NUM_QUBITS])
    u_f(reg[0:NUM_QUBITS], reg[NUM_QUBITS], reg[NUM_QUBITS + 1], reg[NUM_QUBITS + 2])


@qfunc
def check_block(a: QNum, res: QBit):
    res ^= a == 0


@qfunc
def u_amp(
    x: QNum,
    a1: QNum,
    a2: QBit,
    a3: QBit,
):
    # amp = 0.5597575631451602
    AMPLITUDE = get_Amplitude_Gaussian_Fixed(
        MIN, MAX, mean=0.0, sigma=1.0 / np.sqrt(2 * EXP_RATE)
    )
    print("amp:", AMPLITUDE)

    reg = QArray[QBit]("full_reg")
    bind([x, a1, a2, a3], reg)

    exact_amplitude_amplification(
        amplitude=AMPLITUDE,
        oracle=lambda _reg: phase_oracle(
            check_block, _reg[NUM_QUBITS : NUM_QUBITS + 3]
        ),
        space_transform=lambda _reg: state_prep(_reg),
        packed_qvars=reg,
    )

    bind(reg, [x, a1, a2, a3])


@qfunc
def main(
    x: Output[QNum], a1: Output[QNum], a2_qsvt: Output[QBit], a3_qsvt: Output[QBit]
):
    allocate(NUM_QUBITS, x)
    allocate(1, a1)
    allocate(1, a2_qsvt)
    allocate(1, a3_qsvt)
    u_amp(x, a1, a2_qsvt, a3_qsvt)


def parse_qsvt_results(result) -> Dict:
    amps: Dict = {x: [] for x in range(2**NUM_QUBITS)}

    for parsed_state in result.parsed_state_vector:
        # NOTE: amplify가 잘 된다면 "a1"==0, "a2"==0 주석처리하고도 결과 잘 나와야함.
        if (
            parsed_state["a1"] == 0
            and parsed_state["a2_qsvt"] == 0
            and parsed_state["a3_qsvt"] == 0
            and np.linalg.norm(parsed_state.amplitude) > 1e-10
        ):
            amps[parsed_state["x"]].append(parsed_state.amplitude)

    # NOTE: amp에는 amplify에 임시적으로 쓰인 보조 큐비트의 확률진폭도 들어있을 수 있음. (e.g. (x=0, aux=0), (x=0, aux=1))
    simulated_prob = [amp_to_prob(amp) for amp in amps.values()]
    print("np.sum(simulated_prob):", np.sum(simulated_prob))
    # np.sum(simulated_prob): 0.7892557498377792

    # assert np.allclose(np.sum(simulated_prob), 1)
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
    show(qprog)

    x = np.linspace(MIN, MAX, 2**NUM_QUBITS)
    simulated = parse_qsvt_results(result)
    expected = normalize(
        [
            amp_to_prob(F(((MAX - MIN) * x / 2**NUM_QUBITS) + MIN))
            for x in range(2**NUM_QUBITS)
        ]
    )

    plt.plot(x, expected, label="expected")
    plt.plot(x, simulated, label="simulated")
    plt.legend()
    plt.show()

    print("simulated prob:", np.round(simulated, 5))
    print("expected prob:", np.round(expected, 5))

    # assert the probabilities sum to 1
    assert np.allclose(np.sum(simulated), 1)
    assert np.allclose(np.sum(expected), 1)

    # assert the probabilities are close to the ground truth
    assert np.allclose(simulated, expected, atol=1e-2)
    print("PASSED")


if __name__ == "__main__":
    # input amplitude paramter
    parse = argparse.ArgumentParser()
    parse.add_argument("--amp", type=float, default=0.5597575631451602)

    amp = parse.parse_args().amp
    AMPLITUDE = amp
    execute_model()
