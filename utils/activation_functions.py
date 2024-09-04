
import numpy as np


# Funcao sigmoid
def sigmoid_function(layer):
    return 1 / (np.exp(-layer) + 1)


# O layer pode ser a camada escondida ou de saida
# Elas ja estarao pre calculadas com a funcao de ativacao sigmoid
# Derivada da sigmoid pode ser escrita como σ'(x) = σ(x) * (1 - σ(x))
# Considerando σ(x) a funcao sigmoid
def sigmoid_derivative(layer, delta):
    return delta * (layer * (1 - layer))


# funcao da tangente hiperbolica 1 a -1
def tanh_function(layer):
    return (np.exp(layer) - np.exp(-layer)) /\
            (np.exp(layer) + np.exp(-layer))


# derivada da tangente
# ela pode ser escrita como 1 - tanh(x)^2
def tanh_derivative(layer, delta):
    return delta * (1 - np.multiply(layer, layer))


# Funcao de softmax
# Nos proporciona uma distribuicao semelhante a probilistica
# porque obtem a razao em relacao as somas do e^x
# Assim se somarmos eles teremos 1 como resultado por linha
def softmax_function(layer):
    exp_matrix = np.exp(layer)
    return (exp_matrix / exp_matrix.sum(axis=1, keepdims=True))


# Derivada da softmax
# O layer que sera aplicado nesse caso vai ser a camada de saida
# Como essa camada ja esta aplicada a funcao do softmax ou seja
# ja sesta pre calculada nao ha a necessidade de computar o softmax aqui
# a derivada do softmax tem dois casos
# i == j softmax(x) * (1 - softmax(x))
# i != j -softmax(i) * softmax(j)
def softmax_derivative(layer, delta):
    derivative_matrix = []
    for row, dt in zip(layer, delta):
        reshaped_row = row.reshape(-1, 1)
        derivative_matrix.append(
                # Essa subtracao vai nos dar o caso do i == j e i != j
                # porque o diagflat vai fazer uma matriz diagonal
                # com os valores da linha nos dando a operacao para o
                # i == j softmax(i) * (1 - softmax(i)), ja para i != j podemos
                # notar que o sinal de menos
                # no resultado do segundo np.dot no dara o -1 * ( expresssao )
                # e tambem nesse segundo np.dot tambem vai nos dar o produto
                # entre softmax(i) * softmax(j)
                np.dot(np.diagflat(row) - np.dot(reshaped_row,
                                                 reshaped_row.T),
                       dt))
    return np.array(derivative_matrix)
