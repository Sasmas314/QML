import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(thetas):
    # |psi0> подготовка начального состояния
    qml.RY(np.pi / 4, wires=0)
    qml.RY(np.pi / 3, wires=1)
    qml.RY(np.pi / 5, wires=2)

    # Parameterized layer 0
    qml.RZ(thetas[0], wires=0)
    qml.RZ(thetas[1], wires=1)

    # W1 : блок запутывания
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    # Parameterized layer 1
    qml.RY(thetas[2], wires=1)
    qml.RX(thetas[3], wires=2)

    # W2 : второй блок запутывания
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliY(0))         # измеряем <y> на кубите 0 и возвращаем скаляр


thetas0 = np.array([0.1, -0.2, 0.3, 0.4], requires_grad=True)
y = circuit(thetas0)
print(y)
