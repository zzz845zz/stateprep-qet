from typing import Dict
import numpy as np
from classiq import *
from stateprep_qet.utils import amp_to_prob, find_angle, normalize
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin, cos

NUM_QUBITS = 3  # U is applied to these qubits
POLY_DEGREE = 5
POLY_FUNC = lambda x: x
POLY_MAX_SCALE = 1


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def u_sin(x: QNum, a: QNum) -> None:
    a *= sin(x / (2**NUM_QUBITS))  # Amplitude encoding sin(x) to |1>
    X(a)  # sin(x) to |0>


@qfunc
def main(x: Output[QNum], ind: Output[QNum], aux: Output[QBit]):
    allocate(NUM_QUBITS, x)
    allocate(1, ind)
    allocate(1, aux)

    # Construct equal superposition of all states
    hadamard_transform(x)

    # Apply QSVT
    phiset, red_phiset, parity = find_angle(POLY_FUNC, POLY_DEGREE, POLY_MAX_SCALE)
    full_reg = QArray[QBit]("full_reg")
    bind([ind, x], full_reg)
    qsvt(
        phase_seq=phiset,
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
        aux=aux,
    )
    bind(full_reg, [ind, x])


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

    simulated = parse_qsvt_results(result)
    expected = normalize(
        [amp_to_prob(np.sin(x / 2**NUM_QUBITS)) for x in range(2**NUM_QUBITS)]
    )

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
