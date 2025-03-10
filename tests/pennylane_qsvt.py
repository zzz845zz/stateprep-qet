# Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qsvt.html

import pennylane as qml
import numpy as np

# P(x) = -x + 0.5 x^3 + 0.5 x^5
poly = np.array([0, -1, 0, 0.5, 0, 0.5])

hamiltonian = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1) @ qml.Z(2)])

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit():
    qml.qsvt(hamiltonian, poly, encoding_wires=[0], block_encoding="prepselprep")
    return qml.state()


matrix = qml.matrix(circuit, wire_order=[0, 1, 2])()
print(matrix[:4, :4].real)
