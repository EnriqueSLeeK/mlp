
import numpy as np


# Funcao para calcular o erro do modelo inteiro
# E = (Y* - Y)^2 / n
def mean_square_error(output, expected):
    expected[expected == -1] = 0
    return (np.mean(np.power(output - expected, 2))) * 0.5
