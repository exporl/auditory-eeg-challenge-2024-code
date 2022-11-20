"""Default dilation model."""
import tensorflow as tf


def dilation_model(
    time_window=None,
    layers=3,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    compile=True,
    inputs=tuple(),
    output_name="output",
):
    """Convolutional dilation model.

    Parameters
    ----------
    time_window : int or None
        Segment length. If None, the model will accept every time window input
        length.
    layers : int
        Depth of the network/Number of layers
    kernel_size : int
        Size of the kernel for the dilation convolutions
    spatial_filters : int
        Number of parallel filters to use in the spatial layer
    dilation_filters : int
        Number of parallel filters to use in the dilation layers
    activation : str or list or tuple
        Name of the non-linearity to apply after the dilation layers
        or list/tuple of different non-linearities
    compile : bool
        If model should be compiled
    inputs : tuple
        Alternative inputs
    output_name : str
        Name to give to the output
    Returns
    -------
    tf.Model
        The dilation model
    """
    # If different inputs are required
    if len(inputs) == 3:
        eeg, env1, env2 = inputs[0], inputs[1], inputs[2]
    else:
        eeg = tf.keras.layers.Input(shape=[time_window, 64])
        env1 = tf.keras.layers.Input(shape=[time_window, 1])
        env2 = tf.keras.layers.Input(shape=[time_window, 1])

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    env_proj_1 = env1
    env_proj_2 = env2
    # Spatial convolution
    eeg_proj_1 = tf.keras.layers.Conv1D(spatial_filters, kernel_size=1)(eeg)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
        )
        env_proj_1 = env_proj_layer(env_proj_1)
        env_proj_2 = env_proj_layer(env_proj_2)

    # Comparison
    cos1 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_1])
    cos2 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_2])

    # Classification
    out1 = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos1, cos2]))
    )

    # 1 output per batch
    out = tf.keras.layers.Reshape([1], name=output_name)(out1)

    # def print_t(t):
    #     tf.print(t, summarize=-1)
    #     return t
    # out = tf.keras.layers.Lambda(print_t)(out)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["acc"],
            loss=["binary_crossentropy"],
        )
        print(model.summary())
    return model
