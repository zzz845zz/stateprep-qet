from typing import Dict
import numpy as np
from classiq import *
from stateprep_qet.utils import find_angle
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences

NUM_QUBITS = 2  # U is applied to these qubits
DEGREE = 11
FUNC = lambda x: x

INPUT = np.array([0, 1, 0, 0])
UNITARY = np.array(
    [
        [np.sqrt(3 / 16), np.sqrt(13 / 16), 0, 0],
        [np.sqrt(13 / 16), -np.sqrt(3 / 16), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


def get_random_unitary(num_qubits, seed=4):
    np.random.seed(seed)
    X = np.random.rand(2**num_qubits, 2**num_qubits)
    U, s, V = np.linalg.svd(X)

    unitary = U @ V.T
    A_dim = int(unitary.shape[0] / 2)
    A = unitary[:A_dim, :A_dim]
    return unitary


@qfunc
def projector_cnot(reg: QNum, aux: QBit) -> None:
    control(reg == 0, lambda: X(aux))


@qfunc
def main(x: Output[QArray[QBit]], aux: Output[QBit]):
    allocate(1, aux)
    prepare_state(probabilities=INPUT, bound=0.01, out=x)

    phiset, red_phiset, parity = find_angle(FUNC, DEGREE, 1.0)
    qsvt_phases = red_phiset

    qsvt(
        phase_seq=qsvt_phases,
        proj_cnot_1=lambda reg_, aux_: projector_cnot(
            reg_[0], aux_
        ),  # reg==0 representing "from state". If the state is "from state", then mark aux qubit as |1>
        proj_cnot_2=lambda reg_, aux_: projector_cnot(
            reg_[0], aux_
        ),  # reg==0 representing "to state". If the state is "to state", then mark aux qubit as |1>
        u=lambda arg0: unitary(
            elements=UNITARY,
            target=arg0,
        ),
        qvar=x,
        aux=aux,
    )


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

    print("state_vector", result.state_vector)
    print("\nparsed_state_vector")

    d: Dict = {x: [] for x in range(2**NUM_QUBITS)}

    for parsed_state in result.parsed_state_vector:
        if parsed_state["aux"] == 0 and np.linalg.norm(parsed_state.amplitude) > 1e-15:
            print(parsed_state)
            print("prob:", np.abs(parsed_state.amplitude) ** 2)

            x = parsed_state["x"]
            x_int = sum(2**i * x[i] for i in range(len(x)))
            d[x_int].append(parsed_state.amplitude)

    d = {k: np.linalg.norm(v) for k, v in d.items()}
    values = [d[i] for i in range(len(d))]
    x = np.sqrt(np.linspace(0, 1 - 1 / (2**NUM_QUBITS), 2**NUM_QUBITS))
    measured_poly_values = np.sqrt(2**NUM_QUBITS) * np.array(values)

    print(d)
    print(values)
    print(x)
    normalized = measured_poly_values / np.sum(measured_poly_values)
    normalized_prob = np.abs(measured_poly_values) ** 2 / np.sum(
        np.abs(measured_poly_values) ** 2
    )
    print("normalized:", normalized)
    print("normalized prob:", normalized_prob)
    print(measured_poly_values)
    print(INPUT)
    print(UNITARY)
    print(FUNC(UNITARY))
    print(FUNC(UNITARY) @ INPUT)

    # normalized

    assert np.allclose(normalized_prob, FUNC(UNITARY) @ INPUT, atol=1e-1)


if __name__ == "__main__":
    execute_model()
