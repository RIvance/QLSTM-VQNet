import math
import time

import numpy as np
import pandas as pd
import pyvqnet.utils.storage as qml_storage
import pyvqnet.optim as optim
import pyvqnet.nn as nn
from pyvqnet.tensor import QTensor as Tensor
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import *
import matplotlib.pyplot as plt

from QLSTM import QLSTM


def sliding_windows(data, seq_len: int) -> (np.ndarray, np.ndarray):
    xs, ys = [], []
    # x: features from time 0 to t - 1
    # y: features at time t
    for i in range(len(data) - seq_len - 1):
        xs.append(data[i: i + seq_len])
        ys.append(data[i + seq_len])

    return np.array(xs), np.array(ys)


class RegLSTM(nn.Module):
    """
    constructor
    :param input_sz: num of features
    :param hidden_sz: num of hidden neurons
    """

    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.rnn = QLSTM(input_sz, hidden_sz)
        self.regression = nn.Linear(hidden_sz, input_sz)

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        output, (h_n, _) = self.rnn(x)
        # return self.fc(output.transpose(0, 1)[-1])
        output = self.regression(h_n)
        return output


def train(
        model: RegLSTM, learning_rate: float, batch_size: int, num_epochs: int,
        x_train: NDArray, y_train: NDArray, x_test: NDArray, y_test: NDArray,
        data_scaler: MinMaxScaler
) -> RegLSTM:

    print(f"Start training, lr: {learning_rate}, batch_size: {batch_size}, epochs: {num_epochs}")

    criterion = nn.loss.MeanSquaredError()  # mean-squared error for regression
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    losses = []
    loss_history = open('loss.txt', 'a+')

    for epoch_idx in range(num_epochs):

        n_batches = math.ceil(x_train.shape[0] / batch_size)
        print(f"Start training epoch {epoch_idx}, num_batches = {n_batches}")

        start_time_epoch = time.time()

        for batch_idx in range(n_batches):
            optimizer.zero_grad()

            start_time_batch = time.time()

            x_train_batch = Tensor(
                x_train[batch_idx * batch_size: (batch_idx + 1) * batch_size],
                requires_grad=True
            )
            y_train_batch = Tensor(
                y_train[batch_idx * batch_size: (batch_idx + 1) * batch_size],
                requires_grad=True
            )

            y_predict_batch = model(x_train_batch)

            # obtain the loss function
            loss = criterion(y_train_batch, y_predict_batch)
            loss.backward()

            optimizer._step()

            time_consumes_batch = time.time() - start_time_batch

            print(f" - Batch {batch_idx}, train loss: {loss.item()}, took: {int(time_consumes_batch)}")

        y_predict_eval = model(x_test)
        eval_loss = criterion(y_test, y_predict_eval)

        losses.append(eval_loss.item())

        time_consumes_epoch = time.time() - start_time_epoch

        print(f"Epoch: {epoch_idx}, eval loss: {eval_loss.item()}, took: {int(time_consumes_epoch)}")

        y_pred_ndarray = y_predict_eval.data
        y_test_ndarray = y_test.data

        y_pred_origin = data_scaler.inverse_transform(y_pred_ndarray)
        y_test_origin = data_scaler.inverse_transform(y_test_ndarray)

        try:

            for i in range(y_pred_ndarray.shape[1]):
                plt.title(f"epoch {epoch_idx}, feature {i}")
                plt.plot(y_pred_origin[:, i])
                plt.plot(y_test_origin[:, i])
                plt.savefig(f"images/eval_epoch_{epoch_idx}_{i}.png")
                plt.close()

            loss_history.write(f'{eval_loss.item()}\n')

        except:
            # ignore file IO exceptions
            pass

        mdl_id = int(time.time() % 1000000 / 100)
        qml_storage.save_parameters(model.state_dict(), f"models/param_{mdl_id}.model")
        qml_storage.save_parameters(model.state_dict(), f"models/current.model")
        print("model saved")

    plt.plot(np.array(losses))
    plt.title('loss')
    plt.savefig('images/loss.png')
    plt.close()

    return model


if __name__ == "__main__":

    training_set = pd.read_csv('Train.csv')
    testing_set = pd.read_csv('Test.csv')

    training_set = training_set.iloc[:, 1:].values
    testing_set = testing_set.iloc[:, 1:].values

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_set)
    testing_data = scaler.fit_transform(testing_set)

    sequence_length = 4
    x_train, y_train = sliding_windows(training_data, sequence_length)
    x_test, y_test = sliding_windows(testing_data, sequence_length)

    x_train, y_train, x_test, y_test = map(
        lambda x: np.array(x),
        [x_train, y_train, x_test, y_test]
    )

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    # (num, time_seq, features)
    print(x_train.shape)
    print(x_test.shape)

    input_size = x_train.shape[2]
    hidden_size = 4

    # ======== Load Model ========

    lstm = RegLSTM(input_size, hidden_size)

    try:
        params = qml_storage.load_parameters("models/current.model")
        lstm.load_state_dict(params)
        print("Parameters loaded")
    except:
        print("Unable to load param!")

    # ======== Train ========

    train(
        model=lstm, learning_rate=0.025, batch_size=8, num_epochs=5,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        data_scaler=scaler
    )
