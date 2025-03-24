from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from classiq import *
from stateprep_qet.utils import (
    amp_to_prob,
    find_angle,
    h,
    h_hat,
    h_scale,
    l2_norm_filling_fraction,
    normalize,
)
from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences
from classiq.qmod.symbolic import sin, cos, floor


NUM_QUBITS = 8  # resolution of input x
EXP_RATE = 1  # decay rate of the Gaussian

F = lambda x: np.exp(-EXP_RATE * (x**2))  # Gaussian
MIN = -2  # min x
MAX = 2  # max x

H_FUNC = h(f=F, min=MIN, max=MAX)
POLY_FUNC = h_hat(h=H_FUNC, h_max=F(1))
POLY_DEGREE = 15
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
def my_projector_controlled_phase(
    phase_even: CReal,
    phase_odd: CReal,
    proj_cnot: QCallable[QArray[QBit], QBit],
    qvar: QArray[QBit],
    aux: QBit,
    aux2: QBit,
) -> None:
    """
    [Qmod Classiq-library function]

    Assigns a phase to the entire subspace determined by the given projector. Corresponds to the operation:

    $$
    \\Pi_{\\phi} = (C_{\\Pi}NOT) e^{-i\frac{\\phi}{2}Z}(C_{\\Pi}NOT)
    $$

    Args:
        phase: A rotation phase.
        proj_cnot: Projector-controlled-not unitary that sets an auxilliary qubit to |1> when the state is in the projection.
        qvar: The quantum variable to which the rotation applies, which resides in the entire block encoding space.
        aux: A zero auxilliary qubit, used for the projector-controlled-phase rotation. Given as an inout so that qsvt can be used as a building-block in a larger algorithm.
    """
    # within_apply(lambda: proj_cnot(qvar, aux), lambda: RZ(phase_even, aux))
    within_apply(
        lambda: proj_cnot(qvar, aux),
        lambda: control(
            ctrl=(aux2 == 0),
            # TODO: check if this is correct
            stmt_block=lambda: RZ(phase_even, aux),
            else_block=lambda: RZ(phase_odd, aux),
        ),
    )


@qfunc
def my_qsvt_step(
    phase_even1: CReal,
    phase_odd1: CReal,
    phase_even2: CReal,
    phase_odd2: CReal,
    proj_cnot_1: QCallable[QArray[QBit], QBit],
    proj_cnot_2: QCallable[QArray[QBit], QBit],
    u: QCallable[QArray[QBit]],
    qvar: QArray[QBit],
    aux: QBit,
    aux2: QBit,
) -> None:
    u(qvar)
    my_projector_controlled_phase(phase_even1, phase_odd1, proj_cnot_2, qvar, aux, aux2)
    invert(lambda: u(qvar))
    my_projector_controlled_phase(phase_even2, phase_odd2, proj_cnot_1, qvar, aux, aux2)


@qfunc
def my_qsvt(
    phase_even_seq: CArray[CReal],
    phase_odd_seq: CArray[CReal],
    proj_cnot_1: QCallable[QArray[QBit], QBit],
    proj_cnot_2: QCallable[QArray[QBit], QBit],
    u: QCallable[QArray[QBit]],
    qvar: QArray[QBit],
    aux: QBit,
    aux2: QBit,
) -> None:
    print("phase_even_seq.len", phase_even_seq.len)
    print("phase_odd_seq.len", phase_odd_seq.len)
    # assert phase_even_seq.len == phase_odd_seq.len

    H(aux)
    H(aux2)

    my_projector_controlled_phase(
        phase_even_seq[0], phase_odd_seq[0], proj_cnot_1, qvar, aux, aux2
    )
    repeat(
        count=floor((phase_even_seq.len - 1) / 2),
        iteration=lambda index: my_qsvt_step(
            phase_even_seq[2 * index + 1],
            phase_odd_seq[2 * index + 1],
            phase_even_seq[2 * index + 2],
            phase_odd_seq[2 * index + 2],
            proj_cnot_1,
            proj_cnot_2,
            u,
            qvar,
            aux,
            aux2,
        ),
    )

    if_(
        condition=phase_even_seq.len % 2 == 1,
        then=lambda: IDENTITY(qvar),
        else_=lambda: (
            u(qvar),
            my_projector_controlled_phase(
                phase_even_seq[phase_even_seq.len - 1],
                phase_odd_seq[phase_even_seq.len - 1],
                proj_cnot_2,
                qvar,
                aux,
                aux2,
            ),
        ),
    )

    H(aux)
    H(aux2)  # TODO: need this?


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

    # Apply QSVT
    phiset_even, red_phiset_even, parity_even = find_angle(
        POLY_EVEN, POLY_DEGREE, POLY_MAX_SCALE
    )
    phiset_even = np.append(phiset_even, 0)
    phiset_odd, red_phiset_odd, parity_odd = find_angle(
        POLY_ODD, POLY_DEGREE, POLY_MAX_SCALE
    )
    assert parity_even == 0
    assert parity_odd == 1
    print(len(phiset_even), len(phiset_odd))
    assert len(phiset_even) == len(phiset_odd)

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
    # print("state_vector", result.state_vector)
    amps: Dict = {x: [] for x in range(2**NUM_QUBITS)}

    for parsed_state in result.parsed_state_vector:
        if (
            parsed_state["a1"] == 0
            and parsed_state["a2_qsvt"] == 0
            # and parsed_state["a3_qsvt"] == 0  # TODO: check if this is correct
            and np.linalg.norm(parsed_state.amplitude) > 1e-10
        ):
            # print(parsed_state)
            # print("prob:", np.abs(parsed_state.amplitude) ** 2)
            amps[parsed_state["x"]].append(parsed_state.amplitude)

    # simulated_prob = normalize([amp_to_prob(amp) for amp in amps.values()])
    # simulated_prob = normalize([np.imag(amp) for amp in amps.values()])
    # TODO: Fix. e.g. (x=0, aux=0), (x=0, aux=1)에 x=0의 amplitude가 같이 들어있음. 이 경우 x=0의 amplitude를 어케 해석해야하나?
    simulated_prob = [np.imag(amp) for amp in amps.values()]
    print(simulated_prob)
    simulated_prob2 = [amp_to_prob(amp) for amp in amps.values()]
    print("sum", np.sum(simulated_prob2))
    fraction = l2_norm_filling_fraction(H_FUNC, 2**NUM_QUBITS, MIN, MAX)
    print("fraction", fraction)
    print("(0.5*fraction)^2", (0.5 * fraction) ** 2)
    # print("simulated_prob", simulated_prob)
    # assert np.allclose(np.sum(simulated_prob), 1)
    return simulated_prob

    # nf = np.sqrt(np.sum([amp_to_prob(amp) for amp in amps.values()]))
    # simulated_amp = [nf * amp for amp in amps.values()]
    # assert np.allclose(np.sum([np.abs(amp) ** 2 for amp in simulated_amp]) / nf, 1)
    # return simulated_amp


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

    x = np.linspace(MIN, MAX, 2**NUM_QUBITS)
    simulated = parse_qsvt_results(result)
    expected = normalize([amp_to_prob(F(xval)) for xval in x])

    plt.plot(x, expected, label="expected")
    plt.plot(x, simulated, label="simulated")

    # TODO: expected_amp?
    # expected_amp = [(F(xval)) for xval in x]
    # plt.plot(x, expected_amp, label="expected_amp")

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
