"""Code to construct the VLAAI network.
Code was extrcted from https://github.com/exporl/vlaai
"""
import tensorflow as tf


def extractor(
    filters=(256, 256, 256, 128, 128),
    kernels=(64,) * 5,
    dilation_rate = 1,
    input_channels=64,
    normalization_fn=lambda x: tf.keras.layers.LayerNormalization()(x),
    activation_fn=lambda x: tf.keras.layers.LeakyReLU()(x),
    name="extractor",
):
    """Construct the extractor model.

    Parameters
    ----------
    filters: Sequence[int]python 
        Number of filters for each layer.
    kernels: Sequence[int]
        Kernel size for each layer.
    input_channels: int
        Number of EEG channels in the input
    normalization_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to normalize the contents of a tensor.
    activation_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to apply an activation function to the contents of a tensor.
    name: str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The extractor model.
    """
    eeg = tf.keras.layers.Input((None, input_channels))

    x = eeg

    if len(filters) != len(kernels):
        raise ValueError("'filters' and 'kernels' must have the same length")

    # Add the convolutional layers
    i = 0
    for filter_, kernel in zip(filters, kernels):
        i +=1

        if i == len(filters) :
            padding = 'valid'
        else:
            padding = 'valid'
        x = tf.keras.layers.Conv1D(filter_, kernel, dilation_rate=dilation_rate,padding=padding )(x)
        x = normalization_fn(x)
        x = activation_fn(x)
        x = tf.keras.layers.ZeroPadding1D((0, kernel - 1))(x)

    return tf.keras.models.Model(inputs=[eeg], outputs=[x], name=name)


def output_context(
    filter_=64,
    kernel=64,
    input_channels=64,
    normalization_fn=lambda x: tf.keras.layers.LayerNormalization()(x),
    activation_fn=lambda x: tf.keras.layers.LeakyReLU()(x),
    name="output_context_model",
):
    """Construct the output context model.

    Parameters
    ----------
    filter_: int
        Number of filters for the convolutional layer.
    kernel: int
        Kernel size for the convolutional layer.
    input_channels: int
        Number of EEG channels in the input.
    normalization_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to normalize the contents of a tensor.
    activation_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to apply an activation function to the contents of a tensor.
    name: str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The output context model.
    """
    inp = tf.keras.layers.Input((None, input_channels))
    x = tf.keras.layers.ZeroPadding1D((kernel - 1, 0))(inp)
    x = tf.keras.layers.Conv1D(filter_, kernel)(x)
    x = normalization_fn(x)
    x = activation_fn(x)
    return tf.keras.models.Model(inputs=[inp], outputs=[x], name=name)


def vlaai(
    nb_blocks=4,
    extractor_model=None,
    output_context_model=None,
    use_skip=True,
    input_channels=64,
    output_dim=1,
    name="vlaai",
):
    """Construct the VLAAI model.

    Parameters
    ----------
    nb_blocks: int
        Number of repeated blocks to use.
    extractor_model: Callable[[tf.Tensor], tf.Tensor]
        The extractor model to use.
    output_context_model: Callable[[tf.Tensor], tf.Tensor]
        The output context model to use.
    use_skip: bool
        Whether to use skip connections.
    input_channels: int
        Number of EEG channels in the input.
    output_dim: int
        Number of output dimensions.
    name: str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The VLAAI model.
    """
    if extractor_model is None:
        extractor_model = extractor()
    if output_context_model is None:
        output_context_model = output_context()

    eeg = tf.keras.layers.Input((None, input_channels))

    # If using skip connections: start with x set to zero
    if use_skip:
        x = tf.zeros_like(eeg)
    else:
        x = eeg

    # Iterate over the blocks
    for i in range(nb_blocks):
        if use_skip:
            x = extractor_model(eeg + x)
        else:
            x = extractor_model(x)
        x = tf.keras.layers.Dense(input_channels)(x)
        x = output_context_model(x)

    x = tf.keras.layers.Dense(output_dim)(x)

    return tf.keras.models.Model(inputs=[eeg], outputs=[x], name=name)


def pearson_tf(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in tensorflow.

    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    tf.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = tf.reduce_mean(y_true, axis=axis, keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=axis, keepdims=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = tf.reduce_sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        axis=axis,
        keepdims=True,
    )
    std_true = tf.reduce_sum(tf.square(y_true - y_true_mean), axis=axis, keepdims=True)
    std_pred = tf.reduce_sum(tf.square(y_pred - y_pred_mean), axis=axis, keepdims=True)
    denominator = tf.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return tf.reduce_mean(tf.math.divide_no_nan(numerator, denominator), axis=-1)


@tf.function
def pearson_loss(y_true, y_pred, axis=1):
    """Pearson loss function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson loss.
        Shape is (batch_size, 1, n_features)
    """
    return -pearson_tf(y_true, y_pred, axis=axis)


@tf.function
def pearson_metric(y_true, y_pred, axis=1):
    """Pearson metric function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson metric.
        Shape is (batch_size, 1, n_features)
    """
    return pearson_tf(y_true, y_pred, axis=axis)
