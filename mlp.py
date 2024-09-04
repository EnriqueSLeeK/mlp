
import utils.activation_functions as act_f
import utils.confusion_matrix as conf_m
import utils.cost_function as cost_f
import utils.file_operation as fop
import matplotlib.pyplot as plt
import numpy as np


class mlp:

    # Inicializacao dos parametros
    def __init__(self,
                 input_data, input_size,
                 hidden_size, output_size,
                 hidden_activation=act_f.sigmoid_function,
                 hidden_derivative=act_f.sigmoid_derivative,
                 output_activation=act_f.sigmoid_function,
                 output_derivative=act_f.sigmoid_derivative,
                 cost_function=cost_f.mean_square_error):

        # Storing the configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # A simple initialization of variables and data
        if input_size != 0:
            self.input_layer = input_data[:, :input_size]
            self.expected = input_data[:, input_size:]
            self.hidden_layer = np.array([])
            self.output_layer = np.array([])
            self.error = []

            # Weights
            self.hidden_weights = np.random.uniform(
                    size=(input_size, hidden_size))
            self.output_weights = np.random.uniform(
                    size=(hidden_size, output_size))

            # Bias
            self.hidden_bias = np.random.uniform(
                    size=(1, hidden_size))
            self.output_bias = np.random.uniform(
                    size=(1, output_size))

            self.conf_matrix = conf_m.confusion_matrix(self.output_size)

        self.h_act = hidden_activation
        self.h_dv = hidden_derivative

        self.o_act = output_activation
        self.o_dv = output_derivative

        self.cost_function = cost_function

    # Combinacao linear
    def linear_combination(self, weights, features, bias):
        return (features @ weights) + bias

    # Processo de feedforward
    # Os outputs das camadas anteriores alimentam as camadas posteriores
    def feed_forward(self):
        # Primeiro se faz a combinacao linear dos pesos da camada escondida
        # com a camade entrada
        # Depois se aplica a funcao de ativacao
        self.hidden_layer = self.h_act(
                self.linear_combination(self.hidden_weights,
                                        self.input_layer,
                                        self.hidden_bias)
                )
        # Mesmo procedimento
        # Usando os pesos da camada de saida com a camada escondida
        self.output_layer = self.o_act(
                self.linear_combination(self.output_weights,
                                        self.hidden_layer,
                                        self.output_bias)
                )

    # Treinamento
    def train(self, learning_rate=0.2, epoch=500):

        for curr in range(epoch + 1):
            # Feed forward
            self.feed_forward()

            # Calculo do error vai ser usado o mean square error
            # para fazer o calculo
            err = self.cost_function(self.output_layer,
                                     self.expected)
            self.error.append(err)
            ##########

            # Atualizacao dos pesos output
            # delta_output pega a diferenca entre o output e o esperado
            # No caso o delta_output seria um pedaco no calculo da derivada
            # Do erro em relacao aos pesos, ela calculada separadamente
            # porque ela vai ser usada em outros lugares tambem
            delta_output = self.o_dv(self.output_layer,
                                     self.output_layer - self.expected)
            # Calculamos os gradientes aqui
            # usando o resultado do delta output e fazendo uma
            # multiplicacao de matriz
            output_gradients = self.hidden_layer.T @ delta_output

            # Aqui ocorre a atualizacao dos pesos da camada de saida
            self.output_weights -= output_gradients * learning_rate
            ##########

            # Atualizacao dos pesos hidden
            # O delta aqui e um pedaco para o calculo da gradiente
            #
            # O delta_output e usado aqui porque em um pedaco
            # do calculo da gradiente da camada escondida
            delta_hidden = self.h_dv(self.hidden_layer,
                                     delta_output @ self.output_weights.T)
            # Usando o resultado do delta_hidden podemos obter a gradiente
            # da camada escondida
            hidden_gradients = self.input_layer.T @ delta_hidden

            # Aqui ocorre a atualizacao dos pesos da camada escondida
            self.hidden_weights -= hidden_gradients * learning_rate
            ##########

            # Atualizacao dos bias output
            self.output_bias -= np.sum(delta_output,
                                       axis=0,
                                       keepdims=True) *\
                learning_rate
            ##########

            # Atualizacao dos bias hidden
            self.hidden_bias -= np.sum(delta_hidden,
                                       axis=0,
                                       keepdims=True) *\
                learning_rate
            ##########

            # Update da matrix de confusao
            self.conf_matrix.update_matrix(self.expected.argmax(axis=1),
                                           self.output_layer.argmax(axis=1))

            if curr % 500 == 0:
                print(f"{curr}/{epoch} -- ", end=" ")
                self.print_data()

    def print_data(self):
        print(f"Accuracy: {self.get_accuracy(self.classify()) * 100}%",
              end=" ")
        err = self.cost_function(self.output_layer,
                                 self.expected)
        print(f"Loss: {err}")

    # Compara o que foi obtido com o esperado e tira a razao
    # para obter a proporcao de acertos
    def get_accuracy(self, classified):
        size = len(self.input_layer)
        point = 0
        for out, expect in zip(classified, self.expected):
            if (out == expect).all():
                point += 1
        return (float(point) / float(size))

    # Ele vai achar o valor maximo e substitui-lo pelo label positivo
    # no caso seria o 1 e para o labels negativos para zero
    def classify(self):
        classified = np.zeros_like(self.output_layer)
        classified[np.arange(len(self.output_layer)),
                   self.output_layer.argmax(1)] = 1

        return classified

    # Um setter para os pesos e bias e o tamanho da entrada
    def set_weight_bias(self,
                        input_size,
                        h_w,
                        h_b,
                        o_w,
                        o_b):
        self.input_size = input_size
        self.hidden_weights = h_w
        self.hidden_bias = h_b
        self.output_weights = o_w
        self.output_bias = o_b

    # Funcao de previsao com feedforward
    def predict(self, input_data=np.array([])):
        self.input_layer = input_data[:, :self.input_size]
        self.expected = input_data[:, self.input_size:]
        self.expected[self.expected == -1] = 0

        self.feed_forward()
        classified = self.classify()
        self.print_data()
        return classified


def main():
    directory = 'characterSet'

    # data = fop.extract_data('caracteres-limpo', directory)
    # data = fop.extract_data('ruido-ruido20', directory)
    # data = fop.extract_data('limpo-ruido20', directory)

    # Juncao dos dados ruido e ruido20
    data = fop.extract_data('limpo-ruido20', directory)
    # data = fop.extract_data('ruido-ruido20', directory)
    print("Dados de treinamento usados limpo e ruido20")

    # O modelo
    # 63 de entrada
    # 12 escondida
    # 7 saida
    model = mlp(data,
                len(data[0]) - 7,
                12,
                7,
                act_f.sigmoid_function,
                act_f.sigmoid_derivative,
                act_f.softmax_function,
                act_f.softmax_derivative)

    # Taxa de aprendizado 0.05
    # Iteracoes 7500 vezes
    model.train(0.05, 7500)

    # test_data = fop.extract_data('caracteres-ruido', 'characterSet')
    # test_data = fop.extract_data('caracteres-ruido20', 'characterSet')
    print("Dados de teste foi usado os caracteres-ruido")
    test_data = fop.extract_data('caracteres-ruido', directory)
    model.predict(test_data)
    # plt.plot(model.error)
    # plt.show()


if __name__ == '__main__':
    main()
