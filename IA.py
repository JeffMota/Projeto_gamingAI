import random
import numpy as np
import os


def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))

def dsigmoid(x):
    return x * (1 - x)

def createMatrix(linha, coluna):
    data = []
    for _ in range(linha):
        arr = []
        for _ in range(coluna):
            arr.append(random.random() * 2 - 1)
        data.append(arr)

    return data

if os.path.exists('config.txt'):
    with open('./config.txt', 'r+') as configFile:
        lines = configFile.read().split('#\n')
        bias = lines[0].split('_')
        weigths = lines[1].split('_')
else:
    print('Arquivo de configuração não encontrado, criando novo arquivo...')
    bias = False
    weigths = False
    with open('./config.txt', 'w') as configFile:
        configFile.write('')

def writeConfig(bias_ih, bias_hh, bias_ho, weigths_ih, weigths_hh, weigths_ho):
    with open('config.txt', 'w') as configFile:
        configFile.writelines([str(bias_ih.tolist()), '_', str(bias_hh.tolist()), '_', str(bias_ho.tolist()), '\n#\n', str(weigths_ih.tolist()), '_', str(weigths_hh.tolist()), '_', str(weigths_ho.tolist())])


# Função que cria a rede
class RedeNeural:
    def __init__(self, i_nodes, h_nodes, hh_nodes, o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.hh_nodes = hh_nodes
        self.o_nodes = o_nodes

        self.bias_ih = np.array(eval(bias[0]) if bias and eval(bias[0]) else createMatrix(self.h_nodes, 1))
        self.bias_hh = np.array(eval(bias[1]) if bias and eval(bias[1]) else createMatrix(self.hh_nodes, 1))
        self.bias_ho = np.array(eval(bias[2]) if bias and eval(bias[2]) else createMatrix(self.o_nodes, 1))

        self.weigths_ih = np.array(eval(weigths[0]) if weigths and eval(weigths[0]) else createMatrix(self.h_nodes, self.i_nodes))
        self.weigths_hh = np.array(eval(weigths[1]) if weigths and eval(weigths[1]) else createMatrix(self.hh_nodes, self.h_nodes))
        self.weigths_ho = np.array(eval(weigths[2]) if weigths and eval(weigths[2]) else createMatrix(self.o_nodes, self.hh_nodes))
        # self.weigths_ih = np.array(criaMatriz(self.h_nodes, self.i_nodes))
        # self.weigths_hh = np.array(criaMatriz(self.hh_nodes, self.h_nodes))
        # self.weigths_ho = np.array(criaMatriz(self.o_nodes, self.hh_nodes))

    def train(self, arr, target):

        self.learning_rate = 0.1
        input = np.transpose(arr)

        # Input para Hidden
        hidden = np.dot(self.weigths_ih, input)
        hidden = np.add(hidden, self.bias_ih)
        hidden = sigmoid(hidden)

        # Hidden para Hidden 2
        hidden2 = np.dot(self.weigths_hh, hidden)
        hidden2 = np.add(hidden2, self.bias_hh)
        hidden2 = sigmoid(hidden2)

        # Hidden2 para Output
        output = np.dot(self.weigths_ho, hidden2)
        output = np.add(output, self.bias_ho)
        output = sigmoid(output)

        # Backpropagation

        # Output para Hidden2  DeltaWih = E ** d(s) * lr * oT ------------------------------------
        expected = np.transpose(target)

        output_error = np.subtract(expected, output)
        d_output = dsigmoid(output)

        hidden2_t = np.transpose(hidden2)

        gradient = np.multiply(output_error, d_output)
        gradient = np.multiply(gradient, self.learning_rate)

        # Adição dos bias
        self.bias_ho = np.add(self.bias_ho, gradient)

        # Correção de pesos
        weigths_ho_deltas = np.dot(gradient, hidden2_t)

        self.weigths_ho = np.add(self.weigths_ho, weigths_ho_deltas)

        # Correção pesos da Hidden2 DeltaWih = E_o ** d(o)*lr* entradasT -----------------------

        weigths_ho_t = np.transpose(self.weigths_ho)

        hidden2_error = np.dot(weigths_ho_t, output_error)
        d_hidden2 = dsigmoid(hidden2)
        hidden_t = np.transpose(hidden)

        gradient_hh = np.multiply(hidden2_error, d_hidden2)
        gradient_hh = np.multiply(gradient_hh, self.learning_rate)

        # Adição dos bias
        self.bias_hh = np.add(self.bias_hh, gradient_hh)
        # Correção de pesos
        weigths_hh_deltas = np.dot(gradient_hh, hidden_t)
        self.weigths_hh = np.add(self.weigths_hh, weigths_hh_deltas)

        # Correção pesos da Hidden  DeltaWih = E_o ** d(o)*lr* entradasT -----------------------

        weigths_hh_t = np.transpose(self.weigths_hh)
        hidden_error = np.dot(weigths_hh_t, hidden2_error)

        d_hidden = dsigmoid(hidden)
        input_t = np.transpose(input)

        gradient_h = np.multiply(hidden_error, d_hidden)
        gradient_h = np.multiply(gradient_h, self.learning_rate)

        # Adição dos bias
        self.bias_ih = np.add(self.bias_ih, gradient_h)
        # Correção de pesos
        weigths_ih_deltas = np.dot(gradient_h, input_t)
        self.weigths_ih = np.add(self.weigths_ih, weigths_ih_deltas)

        writeConfig(self.bias_ih, self.bias_hh, self.bias_ho, self.weigths_ih, self.weigths_hh, self.weigths_ho)


    def predict(self, arr):
        input = np.transpose(arr)

        # Input para Hidden
        hidden = np.dot(self.weigths_ih, input)
        hidden = np.add(hidden, self.bias_ih)
        hidden = sigmoid(hidden)

        # Hidden para Hidden 2
        hidden2 = np.dot(self.weigths_hh, hidden)
        hidden2 = np.add(hidden2, self.bias_hh)
        hidden2 = sigmoid(hidden2)

        # Hidden2 para Output
        output = np.dot(self.weigths_ho, hidden2)
        output = np.add(output, self.bias_ho)
        output = sigmoid(output)
        output = np.transpose(output)

        return output


#  Rede Neural
dataset = [
    [[[1]], [[52]], [[400]], [[325]], [[2]], [[5]]],
    [[[1]], [[0]], [[0]], [[1]], [[0]], [[1]]]]
