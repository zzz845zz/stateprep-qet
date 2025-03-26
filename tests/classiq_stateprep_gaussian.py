from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from classiq import *
from stateprep_qet.utils import (
    amp_to_prob,
    find_angle,
    h,
    h_hat,
    normalize,
)
from stateprep_qet.qsvt_mixed_parity import my_qsvt
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin


NUM_QUBITS = 4  # resolution of input x
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


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def u_sin(x: QNum, a: QNum) -> None:
    a *= sin(x / (2**NUM_QUBITS))  # Amplitude encoding sin(x) to |1>
    X(a)  # sin(x) to |0>


@qfunc
def main(
    x: Output[QNum], a1: Output[QNum], a2_qsvt: Output[QBit], a3_qsvt: Output[QBit]
):
    allocate(NUM_QUBITS, x)
    allocate(1, a1)
    allocate(1, a2_qsvt)
    allocate(1, a3_qsvt)

    # Construct equal superposition of all states
    hadamard_transform(x)

    # Find phase angles
    phiset_even = find_angle(
        POLY_EVEN, POLY_DEGREE + 3, POLY_MAX_SCALE
    )  # even에 맞게 even degree
    phiset_odd = find_angle(
        POLY_ODD, POLY_DEGREE + 2, POLY_MAX_SCALE
    )  # odd에 맞게 odd degree

    print(len(phiset_even), len(phiset_odd))
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


def parse_qsvt_results(result) -> Dict:
    amps: Dict = {x: [] for x in range(2**NUM_QUBITS)}

    for parsed_state in result.parsed_state_vector:
        # NOTE: amplify가 잘 된다면 "a1"==0, "a2"==0 주석처리하고도 결과 잘 나와야함.
        if (
            parsed_state["a1"] == 0
            and parsed_state["a2_qsvt"] == 0
            and parsed_state["a3_qsvt"] == 1
            and np.linalg.norm(parsed_state.amplitude) > 1e-10
        ):
            amps[parsed_state["x"]].append(parsed_state.amplitude)
    # simulated_prob = normalize([amp_to_prob(amp) for amp in amps.values()])
    simulated_prob = normalize([amp_to_prob(sum(amp)) for amp in amps.values()])
    print("a3_qsvt=1 (odd):", [amp for amp in amps.values()])

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

    # NOTE:
    # e.g.
    # - (x=0, a3_qsvt=0)에는 x=0 일 때 even polynomial의 amplitude가 들어있음.
    # - (x=0, a3_qsvt=1)에는 x=0 일 때 odd polynomial의 amplitude가 들어있음.
    # => sum(amp) 해준 뒤 처리
    a1 = np.sum([amp for amp in amps.values()])
    print("a1:", a1)
    print("sqrt(a1):", np.sqrt(a1))
    # simulated_prob = normalize([amp_to_prob(amp) for amp in amps.values()])
    simulated_prob = normalize([amp_to_prob(sum(amp)) for amp in amps.values()])
    print("a3_qsvt=0 (even):", [amp for amp in amps.values()])

    # print(
    #     "unnormlaized simulated prob:",
    #     sum([amp_to_prob(sum(amp)) for amp in amps.values()]),
    # )
    # print(
    #     "sqrt(unnormalized simulated prob):",
    #     np.sqrt(sum([amp_to_prob(sum(amp)) for amp in amps.values()])),
    # )
    # a1: 0.23760964441646443
    # sqrt(a1): 0.48745219705778786
    # unnormlaized simulated prob: 0.23760964441646443
    # sqrt(unnormalized simulated prob): 0.48745219705778786
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
    execute_model()
