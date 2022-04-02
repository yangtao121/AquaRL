import tensorflow as tf
from tensorflow.keras import layers


def gaussian_mlp(state_dims, output_dims, hidden_size, hidden_activation='relu', output_activation='tanh',
                 name='gaussian_mlp'):
    state_input_layer = layers.Input(shape=(state_dims,), name=name + '_state_input_layer',
                                     )

    hidden_layer = layers.Dense(hidden_size[0], activation=hidden_activation, name=name + '_hidden_' + str(0),
                                )(
        state_input_layer)

    for i, num in enumerate(hidden_size):
        if i == 0:
            continue
        hidden_layer = layers.Dense(num, activation=hidden_activation, name=name + '_hidden_' + str(i),
                                    )(hidden_layer)

    mu = layers.Dense(output_dims, activation=output_activation, name=name + '_mu',
                      )(hidden_layer)

    sigma = layers.Dense(output_dims, activation='softplus', name=name + '_sigma_output',
                         )(hidden_layer)

    model = tf.keras.Model(inputs=state_input_layer,
                           outputs=[mu, sigma],
                           name=name + '_model', )

    return model


# @tf.function
def mlp(state_dims, output_dims, hidden_size, hidden_activation='relu', output_activation=None,
        name='mlp'):
    state_input_layer = layers.Input(shape=(state_dims,), name=name + '_state_input_layer')

    hidden_layer = layers.Dense(hidden_size[0], activation=hidden_activation, name=name + '_hidden_' + str(0))(
        state_input_layer)

    for i, num in enumerate(hidden_size):
        if i == 0:
            continue
        hidden_layer = layers.Dense(num, activation=hidden_activation, name=name + '_hidden_' + str(i))(hidden_layer)

    output_layer = layers.Dense(output_dims, activation=output_activation)(hidden_layer)

    model = tf.keras.Model(inputs=state_input_layer,
                           outputs=output_layer,
                           name=name + '_model')

    return model