import math
import numpy as np
from pyqpanda import *
import pyvqnet.nn as nn
import pyvqnet.qnn as qnn
import pyvqnet.tensor as tensor
from pyvqnet.qnn.measure import expval
from pyvqnet.nn.activation import *
from pyvqnet.tensor import QTensor as Tensor


def vqc_encoding(qubits: QVec, inputs: Tensor):
    """
    :param qubits: QVec[n_features]
    :param inputs: Array[n_features]
    :return: circuit
    """

    circuit = QCircuit()

    ry_params = np.arctan(inputs)
    rz_params = np.arctan(inputs ** 2)

    for i, (ry_param, rz_param) in enumerate(zip(ry_params, rz_params)):
        circuit << H(qubits[i])
        circuit << RY(qubits[i], ry_param)
        circuit << RZ(qubits[i], rz_param)

    return circuit


def vqc_ansatz(qubits: QVec, params: Tensor):
    """
    :param qubits: QVec[n_features]
    :param params: Array[3, n_features]
    :return: circuit
    """

    circuit = QCircuit()
    n_qubits = len(qubits)

    # Entangling layer
    for i in range(1, 3):
        for j in range(n_qubits):
            circuit << CNOT(qubits[j], qubits[(j + i) % n_qubits])

    circuit << BARRIER(qubits)

    # Variational layer
    for i in range(n_qubits):
        circuit << RX(qubits[i], params[0][i])
        circuit << RY(qubits[i], params[1][i])
        circuit << RZ(qubits[i], params[2][i])

    return circuit


def vqc_circuit(inputs, param: Tensor, qubits, cubits, machine):
    circuit = QCircuit()
    circuit << vqc_encoding(qubits, inputs)
    circuit << BARRIER(qubits)
    circuit << vqc_ansatz(qubits, param.reshape([3, -1]))

    vqc_prog = QProg()
    vqc_prog.insert(circuit)

    n_qubits = len(qubits)

    rlt = [expval(machine, vqc_prog, {f"Z{i}": 1}, qubits) for i in range(n_qubits)]

    return np.array(rlt)


device = 'cpu'


class QLSTM(nn.Module):

    def __init__(self, input_sz, hidden_sz):

        super().__init__()

        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.concat_size = input_sz + hidden_sz
        self.n_qubits = hidden_sz

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        self.W = nn.Parameter((input_sz, hidden_sz * 4))
        self.U = nn.Parameter((hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter((hidden_sz * 4))

        self.vqc_forget = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, device, self.n_qubits)
        self.vqc_input = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, device, self.n_qubits)
        self.vqc_update = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, device, self.n_qubits)
        self.vqc_output = qnn.QuantumLayer(vqc_circuit, self.n_qubits * 3, device, self.n_qubits)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.fill_rand_signed_uniform_(stdv)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.shape

        hidden_seq = []
        if init_states is None:
            h_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True)
            c_t = Tensor(np.zeros((batch_sz, self.hidden_size)), requires_grad=True)
        else:
            h_t, c_t = init_states

        hs = self.hidden_size

        for t in range(seq_sz):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # batch the computations into a single matrix multiplication
            gates = tensor.matmul(x_t, self.W) + tensor.matmul(h_t, self.U) + self.bias

            # input
            i_t = self.sigmoid(self.vqc_input(gates[:, : hs]))
            # forget
            f_t = self.sigmoid(self.vqc_forget(gates[:, hs: hs * 2]))
            g_t = self.tanh(self.vqc_update(gates[:, hs * 2: hs * 3]))
            # output
            o_t = self.sigmoid(self.vqc_output(gates[:, hs * 3:]))
            c_t = f_t * c_t + i_t * g_t

            h_t = o_t * self.tanh(c_t)
            hidden_seq.append(tensor.unsqueeze(h_t, 0))

        hidden_seq = tensor.concatenate(hidden_seq, 0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose([1, 0, 2])

        return hidden_seq, (h_t, c_t)

