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

def packed_loss(predict, target, length, criterion):
	# sort
	sort_idx = torch.sort(-length)[1]
	predict = predict[sort_idx]
	target = target[sort_idx]
	length = length[sort_idx]

	target = pack(target, length, batch_first=True)[0]
	predict = pack(predict, length, batch_first=True)[0]
	loss = criterion(predict, target)
	return loss

def masked_softmax(logits, mask, dim=1, epsilon=1e-5):
	""" logits, mask has same size """
	masked_logits = logits.masked_fill(mask == 0, -1e9)
	max_logits = torch.max(masked_logits, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_logits-max_logits)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	return masked_exps/masked_sums
