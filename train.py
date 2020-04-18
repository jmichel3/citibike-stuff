import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import collections
import os
import argparse
import datetime as dt

from model import Model
from utils import DataLoader

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--skip_load_data', type=bool, default=0, help='skip calling the DataLoader')
	parser.add_argument('--rnn_size', type=int, default=400, help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='nn', help='nn')
	parser.add_argument('--batch_size', type=int, default=50, help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=300, help='RNN sequence length')
	parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
	parser.add_argument('--save_every', type=int, default=500, help='save frequency')
	parser.add_argument('--model_dir', type=str, default='save', help='directory to save model to')
	parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
	parser.add_argument('--num_mixture', type=int, default=20, help='number of gaussian mixtures')
	parser.add_argument('--data_scale', type=float, default=20, help='factor to scale raw data down by')
	parser.add_argument('--keep_prob', type=float, default=0.8, help='dropout keep probability')
	args = parser.parse_args()
	train(args)

	return

def train(args):

	# Load data
	# if not args.skip_load_data:
	data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale)

		# np.save(os.path.join(self.data, "dataLoader"), data)
	# else:
		# print("Skipping data loading...")
		# np.load(data_file)

	# Instantiate model
	# model = Model(args)

	model.input_data

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(2))
	model.add(tf.keras.layers.Activation('softmax'))
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

	model.fit(data_loader.train_data, data_loader.valid_data, epochs=1)
	return

main()


