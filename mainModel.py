from tensorflow.keras import layers
from tensorflow.keras.models import Model

from your_project.tensorflow_utils import custom_residual_block


def create_custom_model(input_dim, output_dim, activation="relu", dropout_rate=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input_data")

    # Normalize input images
    normalized_inputs = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = custom_residual_block(normalized_inputs, 32, activation=activation, skip_connection=True, strides=1, dropout=dropout_rate)

    x2 = custom_residual_block(x1, 32, activation=activation, skip_connection=True, strides=2, dropout=dropout_rate)
    x3 = custom_residual_block(x2, 32, activation=activation, skip_connection=False, strides=1, dropout=dropout_rate)

    x4 = custom_residual_block(x3, 64, activation=activation, skip_connection=True, strides=2, dropout=dropout_rate)
    x5 = custom_residual_block(x4, 64, activation=activation, skip_connection=False, strides=1, dropout=dropout_rate)

    x6 = custom_residual_block(x5, 128, activation=activation, skip_connection=True, strides=2, dropout=dropout_rate)
    x7 = custom_residual_block(x6, 128, activation=activation, skip_connection=True, strides=1, dropout=dropout_rate)

    x8 = custom_residual_block(x7, 128, activation=activation, skip_connection=False, strides=1, dropout=dropout_rate)
    x9 = custom_residual_block(x8, 128, activation=activation, skip_connection=False, strides=1, dropout=dropout_rate)

    reshaped = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    bidirectional_lstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(reshaped)
    bidirectional_lstm = layers.Dropout(dropout_rate)(bidirectional_lstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(bidirectional_lstm)

    custom_model = Model(inputs=inputs, outputs=output)
    return custom_model
