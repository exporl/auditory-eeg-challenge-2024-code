""" This module contains linear backward model"""
import tensorflow as tf

from task2_regression.models.vlaai import pearson_tf


@tf.function
def pearson_loss_cut(y_true, y_pred, axis=1):
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
    return -pearson_tf(y_true[:, : tf.shape(y_pred)[1], :], y_pred, axis=axis)


@tf.function
def pearson_metric_cut(y_true, y_pred, axis=1):
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
    return pearson_tf(y_true[:, : tf.shape(y_pred)[1], :], y_pred, axis=axis)


def simple_linear_model(integration_window=32, nb_filters=1, nb_channels=64):
    inp = tf.keras.layers.Input(
        (
            None,
            nb_channels,
        )
    )
    out = tf.keras.layers.Conv1D(nb_filters, integration_window)(inp)
    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(
        tf.keras.optimizers.Adam(),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model

def simple_linear_model_stimulus(integration_window=32, nb_filters=1, nb_channels=64):
    inp = tf.keras.layers.Input(
        (
            None,
            nb_channels,
        )


    )
    # env = abs(s)
    # f0= np.phase(s)
    # f0 = np.angle(s)

    # reconstruct env
    # reconsturct f0
    # reconstructed s = real(reconstructed_env .*exp(1j*reconstructed_f0))./ np.max(abs(reconstructed_env))

    out = tf.keras.layers.Conv1D(nb_filters, integration_window)(inp)
    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(
        tf.keras.optimizers.Adam(),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model
