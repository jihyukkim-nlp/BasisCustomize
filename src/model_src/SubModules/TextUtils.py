import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

def text_padding(text, length, padding_idx, eos_idx=None, sos_idx=None):
	""" 
	text: list of token indices
	length: list of length of tokens (same size with text)
	return: padded text tokens, ndarray, np.int64
	"""
	maxlen = max(length)
	num_data = len(text)
	_append_length = 0
	st = 0
	if eos_idx is not None: 
		_append_length+=1
	if sos_idx is not None: 
		_append_length+= 1
		st = 1
	if padding_idx:
		padded_sentences = np.zeros((num_data, maxlen+_append_length), dtype=np.int64)
	else:
		padded_sentences = np.zeros((num_data, maxlen+_append_length), dtype=np.int64)+padding_idx
	if sos_idx is not None:
		padded_sentences[:, 0] = sos_idx
	
	if eos_idx is not None:
		for i, (l, x) in enumerate(zip(length, text)):
			padded_sentences[i][st:st+l] = x
			padded_sentences[i][st+l] = eos_idx
	else:
		for i, (l, x) in enumerate(zip(length, text)):
			padded_sentences[i][st:st+l] = x
		
	return padded_sentences
