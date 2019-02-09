import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pickle
import os

class InformationCenter():
	def __init__(self, model_type):
		self.model_type = model_type
		super(InformationCenter, self).__init__()

		# Data Loader
		data_dir = './processed_data'
		usr_train, prd_train, x_train, x_train_lens, y_train, \
		usr_dev, prd_dev, x_dev, x_dev_lens, y_dev, \
		usr_test, prd_test, x_test, x_test_lens, y_test, \
		u_dict, p_dict, usr_freq, prd_freq, x_dict, x_vectors = pickle.load(open(data_dir + '/flat_data.p', 'rb'))
		self.word_vectors = x_vectors
		self.train_data = np.array(list(zip(x_train, x_train_lens, usr_train, prd_train, y_train)))
		self.dev_data = np.array(list(zip(x_dev, x_dev_lens, usr_dev, prd_dev, y_dev)))
		self.test_data = np.array(list(zip(x_test, x_test_lens, usr_test, prd_test, y_test)))

		# Meta Information: Given
		self.word2idx = x_dict
		self.user2idx = u_dict
		self.product2idx = p_dict

		# Meta Information: Inferenced
		self.PADDING = "<PAD>"
		self.UNKNOWN = "<UNK>"
		self.idx2word = {self.word2idx[word]:word for word in self.word2idx.keys()}
		self.vocab_size = len(self.word2idx)
		self.num_usr = len(self.user2idx)
		self.num_prd = len(self.product2idx)
		self.num_label = 5

		# Model Hyper-Parameters
		self.emb_size = 300
		self.state_size = 256
		self.query_size = 64
		self.num_attribute = 2 # user and product
		self.key_size = self.query_size * self.num_attribute 
		
		# Training Configuration
		self.BATCH_SIZE = 32
		self.VALID_STEP = 1000 # evaluatino with development set every 1000 update 
		self.EPOCH = 10
		
	def to_var(self, x):
		return Variable(x).cuda() if torch.cuda.is_available() else Variable(x)

	def to_var_fixed(self, x):
		return Variable(x, requires_grad=False).cuda() if torch.cuda.is_available() else Variable(x, requires_grad=False)

	def to_numpy(self, x):
		return x.cpu().data.numpy()

	def split_data(self, batch, sorting):
		(x_batch, x_lens_batch, 
		usr_batch, prd_batch,
		y_batch) = zip(*batch)

		x_batch = np.array(x_batch)
		x_lens_batch = np.array(x_lens_batch)
		usr_batch = np.array(usr_batch)
		prd_batch = np.array(prd_batch)
		y_batch = np.array(y_batch)
		
		if sorting:
			sorted_idx 		= np.argsort(-x_lens_batch)
			x_batch 		= x_batch 		[sorted_idx]
			x_lens_batch 	= x_lens_batch 	[sorted_idx]
			usr_batch 		= usr_batch 	[sorted_idx]
			prd_batch 		= prd_batch 	[sorted_idx]
			y_batch 		= y_batch 		[sorted_idx]

		return (x_batch, x_lens_batch,
				usr_batch, prd_batch,
				y_batch)

	def add_pad(self, sentences, lengths):
		maxlen = np.max(lengths)
		padded_sentences = []
		for i, x in enumerate(sentences):
			if lengths[i] < maxlen : padded_sentences.append([int(w) for w in x]+[0]*(maxlen-lengths[i]))
			else: padded_sentences.append([int(w) for w in x])
		return padded_sentences