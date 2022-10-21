import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
layers = keras.layers
tfd = tfp.distributions
import collections
from collections import defaultdict
import os


class LSTMAndWindowCell(keras.layers.Layer):
  def __init__(self, units, alphabet_size, windows, return_mu=False, rnn_config={}, RNNCellType=keras.layers.LSTMCell, **kwargs):
    super(LSTMAndWindowCell, self).__init__(**kwargs)
    self.units = units
    self.output_size = alphabet_size+units
    self.n_windows = windows
    self.internal_cell = RNNCellType(units, **rnn_config)
    self.alphabet_size = alphabet_size
    self.state_size  = [tf.TensorShape((windows,1)), alphabet_size] + [units, units]
    self.dense = tf.keras.layers.Dense(3*self.n_windows)
    self.return_mu = return_mu

  def build(self, input_shape):
    cell_input_shape = list(input_shape)
    cell_input_shape[-1] += self.alphabet_size
    self.internal_cell.build(cell_input_shape)
    self.dense.build((self.units))
    layers = [self.internal_cell, self.dense]
    self._trainable_weights = [j for i in layers for j in i.trainable_weights]
    super(LSTMAndWindowCell, self).build(input_shape)

  def call(self, inputs, states, constants):
    xyl = inputs
    transcription = constants[0]
    previous_mu, previous_window = states[:2]
    rnn_state = states[2:]

    cell_out, rnn_state = self.internal_cell.call(tf.concat((xyl, previous_window), -1), rnn_state)
    rnn_state = list(rnn_state)

    window_params = self.dense(cell_out)

    d_mu = window_params[..., :self.n_windows, None]
    d_mu = keras.activations.exponential(d_mu)
    mu = d_mu + previous_mu

    sigma = keras.activations.exponential(window_params[..., self.n_windows:2*self.n_windows, None])

    pi = keras.activations.exponential(window_params[..., 2*self.n_windows:, None])

    lin = tf.range(0., tf.shape(transcription)[-2])

    phi = tf.reduce_sum(pi*tf.exp(-sigma*(mu-lin)**2), -2, keepdims=True)

    w = tf.matmul(phi, transcription)
    w = tf.squeeze(w, -2)

    output = [tf.concat((cell_out, w), -1), mu] if self.return_mu else tf.concat((cell_out, w), -1)

    return output, [mu, w] + rnn_state


def get_mixture_loss_func(n_mixtures, n_dims=2):
  def get_loss(true, mix_params):
    n_components = mix_params.shape[-1]
    
    mu = mix_params[...,:n_dims*n_mixtures]
    mu = tf.reshape(mu, tf.concat([tf.shape(mu)[:-1], [n_mixtures, n_dims]],-1))
    sigma = mix_params[...,n_dims*n_mixtures:2*n_dims*n_mixtures]
    sigma = tf.reshape(sigma, tf.concat([tf.shape(sigma)[:-1], (n_mixtures, n_dims)], -1))
    logits = mix_params[...,2*n_dims*n_mixtures:]

    cat = tfd.Categorical(logits=logits)
    comp = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    mixture = tfd.MixtureSameFamily(
        mixture_distribution=cat,
        components_distribution=comp)

    loss = - tf.reduce_mean(mixture.log_prob(true))

    return loss

  return get_loss


def handwriting_synthesis_nn(
    batch_size = None,
    alphabet_size = 81,
    seq_len = 128,
    stroke_len=None,
    n_mixtures = 16,
    n_rnn_neurons = 512,
    n_windows = 16,
    stateful = False,
    increment_window=False,
    conditional=False,
    return_model=True,
    output_window_mu=False,
    use_GRU=False):

  transcription_input = tf.keras.Input((seq_len, alphabet_size), batch_size=batch_size)
  xyl_input = tf.keras.Input((stroke_len, 3), batch_size=batch_size)
  previous_mu = tf.keras.Input((seq_len,n_windows,1), batch_size=batch_size)


  rnn_config = dict(
      kernel_regularizer=keras.regularizers.l2(),
      recurrent_regularizer=keras.regularizers.l2()
  )

  if use_GRU:
    RNNType = keras.layers.GRU
    RNNCellType = keras.layers.GRUCell
  else:
    RNNType = keras.layers.LSTM
    RNNCellType = keras.layers.LSTMCell

  if conditional:
    cell = LSTMAndWindowCell(n_rnn_neurons, alphabet_size, n_windows, rnn_config=rnn_config, RNNCellType=RNNCellType)
    window_out = keras.layers.RNN(cell, return_sequences=True, stateful=stateful, name="rnn1")(xyl_input, constants=transcription_input)
    if output_window_mu:
      window_layer = window_out[0]
      mu = window_out[1]
    else:
      window_layer = window_out
  else:
    window = keras.backend.zeros( (xyl_input.shape[0], xyl_input.shape[1], alphabet_size) )

    window = keras.layers.GaussianNoise(1)(window)
    hidden1 = RNNType(n_rnn_neurons, return_sequences=True, stateful=stateful, name="rnn1", **rnn_config)(tf.keras.layers.Concatenate(-1)([xyl_input, window]))
    window_layer = keras.layers.Concatenate(-1)([hidden1, window])

  merged1 = tf.keras.layers.Concatenate()([xyl_input, window_layer])
  hidden2 = RNNType(n_rnn_neurons, return_sequences=True, stateful=stateful, name="rnn2", **rnn_config)(merged1)
  hidden2 = RNNType(n_rnn_neurons, return_sequences=True, stateful=stateful, name="rnn3", **rnn_config)(hidden2)

  xy_mu = keras.layers.Dense(2*n_mixtures, name="dense1")(hidden2)
  xy_sigma = keras.layers.Dense(2*n_mixtures, activation=keras.activations.exponential, name='dense2')(hidden2)
  xy_logits = keras.layers.Dense(n_mixtures, name='dense3')(hidden2)
  xy_output = keras.layers.Concatenate()([xy_mu, xy_sigma, xy_logits])

  l_output   = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='dense4')(hidden2)

  xyl_output = tf.keras.layers.Concatenate()([xy_output, l_output])

  inputs = [xyl_input, transcription_input]
  outputs =  [xyl_output, mu] if output_window_mu else xyl_output
  if return_model:
    return tf.keras.Model(inputs=inputs, outputs=outputs)
  else:
    return inputs, outputs


