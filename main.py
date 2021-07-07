# Required packages: numpy tensorflow>=2.0 tensorflow_probability tensorflow-addons svgwrite

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
layers = keras.layers
tfd = tfp.distributions
import collections
from collections import defaultdict
import os
from tensorflow.python.client import device_lib
from model import utils, model

## For interactive mode
# import matplotlib.pyplot as plt
# EAGER = True
# if not EAGER:
#   tf.compat.v1.disable_eager_execution()
# import html


if __name__=="__main__":
	data_dir = "original_part/"
	model_dir = "model/"

	# Import utils.py
	with tf.device("/cpu:0"):
	  exec(open(model_dir+"utils.py").read())
  		DataLoader(data_dir=data_dir).preprocess(data_dir, "strokes_training_data.cpkl", "transcriptions.cpkl")

	dataloader = DataLoader(data_dir=".")
	[x_batch, transcriptions_batch], y_batch = dataloader.__getitem__(2)
	x_batch.shape
	y_batch.shape



	try:
	  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
	  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

	  if EAGER:
	    tf.config.experimental_connect_to_cluster(tpu)
	  tf.tpu.experimental.initialize_tpu_system(tpu)
	  strategy = tf.distribute.experimental.TPUStrategy(tpu)

	  print("REPLICAS: ", strategy.num_replicas_in_sync)
	except:
	  strategy = tf.distribute.get_strategy()


	batch_size=64
	seq_len = 1024
	n_rnn_neurons=256
	n_mixtures=16
	n_windows=8


	dataloader = DataLoader(data_dir=".", batch_size=batch_size, maximise_seq_len=True, seq_length=512)
	max_transcription_length = 16

	with strategy.scope():
	  mix_loss = get_mixture_loss_func(n_mixtures, 2)

	  def handwriting_loss(xyl_true, xyl_pred):
	    l_loss  = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(xyl_true[..., -1], xyl_pred[..., -1]))
	    xy_loss = mix_loss(xyl_true[..., :-1], xyl_pred[..., :-1])
	    return l_loss+xy_loss

	  model = handwriting_synthesis_nn(batch_size=batch_size, 
	    n_rnn_neurons=n_rnn_neurons,
	    n_mixtures=n_mixtures, 
	    seq_len=max_transcription_length, 
	    stroke_len=seq_len, 
	    n_windows=n_windows,
	    stateful=False, 
	    conditional=False,
	    use_GRU=False)
	  model.compile(optimizer=keras.optimizers.RMSprop(), loss=handwriting_loss, run_eagerly=False)

	model_unconditional = model


	exec(open(model_dir+"utils.py").read())

	total_samples = 1482 - (1482%batch_size)
	dataloader = DataLoader(data_dir=".", batch_size=total_samples, maximise_seq_len=True, seq_length=seq_len)
	x, y = dataloader.__getitem__(0)
	x = [a.numpy() for a in x]
	x[1] = x[1][:,:max_transcription_length]
	y = y.numpy()


	model.optimizer.learning_rate = 0.001
	model.fit(x, y,
	          batch_size=batch_size,
	          epochs=100,
	          verbose=2)


	model.optimizer.learning_rate = 0.0001
	model.fit(x, y,
	          batch_size=batch_size,
	          epochs=200,
	          verbose=2)


	model.optimizer.learning_rate = 0.00005
	model.fit(x, y,
	          batch_size=batch_size,
	          epochs=200,
	          verbose=2)


	model.optimizer.learning_rate = 0.00002

	model.fit(x, y,
	          batch_size=batch_size,
	          epochs=200,
	          verbose=2)