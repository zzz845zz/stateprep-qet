from classiq import *
from classiq.qmod.symbolic import sin, cos, floor


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
                phase_odd_seq[phase_odd_seq.len - 1],
                # 0,
                proj_cnot_2,
                qvar,
                aux,
                aux2,
            ),
        ),
    )

    H(aux)
    H(aux2)  # TODO: need this?
