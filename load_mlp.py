
import utils.file_operation as fop
import matplotlib.pyplot as plt
import utils.activation_functions as act_f
import utils.create_dict as c_dict
import mlp

data_dir = 'characterSet'
param_dir = "exportedParams"

hyper_param = c_dict.extract_dict(param_dir, "hyper_param.txt")


def preprocess_run():

    model = mlp.mlp(None,
                    0,
                    hyper_param['camada_escondida'],
                    hyper_param['camada_saida'],
                    act_f.sigmoid_function,
                    act_f.sigmoid_derivative,
                    act_f.softmax_function,
                    act_f.softmax_derivative
                    )

    model.set_weight_bias(
            hyper_param['camada_entrada'],
            fop.extract_data('final_hidden_weight', param_dir),
            fop.extract_data('final_hidden_bias', param_dir),
            fop.extract_data('final_output_weight', param_dir),
            fop.extract_data('final_output_bias', param_dir)
            )

    model.predict(fop.extract_data("caracteres-ruido", data_dir))


def train_with_predetermined_param():

    model = mlp.mlp(fop.extract_data("limpo-ruido20", data_dir),
                    hyper_param['camada_entrada'],
                    hyper_param['camada_escondida'],
                    hyper_param['camada_saida'],
                    act_f.sigmoid_function,
                    act_f.sigmoid_derivative,
                    act_f.softmax_function,
                    act_f.softmax_derivative
                    )

    model.train(0.05, 7500)

    model.predict(fop.extract_data("caracteres-ruido", data_dir))


# Para testar o modelo ja treinado com os parametros no e-disciplinas
# Descomentar preprocess_run()
# Para treinar com os parametros iniciais
# Descomentar train_with_predetermined_param()
if __name__ == "__main__":
    # preprocess_run()
    train_with_predetermined_param()
    pass
