from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 256, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 256, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 512, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 512, activation=activation, skip_conv=True, strides=1, dropout=dropout) #1024 maybe


    squeezed = layers.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)

    blstm1 = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(squeezed)
    blstm1 = layers.Dropout(dropout)(blstm1)

    blstm2 = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(blstm1)
    blstm2 = layers.Dropout(dropout)(blstm2)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm2)

    model = Model(inputs=inputs, outputs=output)
    return model
