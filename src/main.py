import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model_src.BiLSTM import BiLSTM
from model_src.cust import cust
from model_src.basis_cust import basis_cust
from InformationCenter import InformationCenter
from LoggingModule import LoggingModule


parser = argparse.ArgumentParser()
baseline_models = ['BiLSTM']
cust_models = ['word_cust', 'encoder_cust', 'attention_cust', 'linear_cust', 'bias_cust']
basis_cust_models = ['word_basis_cust', 'encoder_basis_cust', 'attention_basis_cust', 'linear_basis_cust', 'bias_basis_cust']
model_choices = baseline_models + cust_models + basis_cust_models
parser.add_argument("model_type", choices=model_choices, help="Give model type.")
parser.add_argument("--num_bases", type=int, default=0)

args = parser.parse_args()
if 'basis' in args.model_type and args.num_bases==0:
	print(" must input number of bases (\"--num_bases\") for basis_cust model type")
	print(" e.g. python main.py word_basis_cust --num_bases 3")
	exit()




class modelClassifier:
	def __init__(self):
		# Information Center
		self.information_center = InformationCenter(model_type=args.model_type)
		if 'basis_cust' in self.information_center.model_type:
			self.information_center.num_bases = args.num_bases
		# Logging Module
		self.logger = LoggingModule()
		self.logger.train_loss = self.logger.loss_trace_module()
		self.logger.dev_loss = self.logger.performance_trace_module(higher_is_better=False)
		self.logger.dev_acc = self.logger.performance_trace_module(higher_is_better=True)
		self.logger.dev_rmse = self.logger.performance_trace_module(higher_is_better=False)

		# MODEL DECLARATION
		self.model = self.model_declaration()

		# LOAD PRETRAIN WORD EMBEDDING MATRIX : GLOVE
		pretrained_wordvectors = self.information_center.word_vectors
		pretrained_wordvectors[0] = np.zeros(300) # zero vector for padding(<PAD>)
		self.model.embed.weight.data.copy_(torch.from_numpy(pretrained_wordvectors))
		print("pretrained embedding matrix loaded .. ")
		
		# OPTIMIZER DECLARATION
		parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optimizer = torch.optim.Adadelta(parameters, lr=1.0, rho=0.9, eps=1e-6)
		self.criterion = F.cross_entropy

		# PARAM SAVE DIRECTORY
		self.param_dir = './save_param/'
		if not os.path.exists(self.param_dir): os.mkdir(self.param_dir) # ./save_param
		self.param_dir = os.path.join(self.param_dir, self.information_center.model_type)
		if 'basis_cust' in self.information_center.model_type: 
			self.param_dir += '({}).pth'.format(self.information_center.num_bases)
		else:
			self.param_dir += '.pth'

	def model_declaration(self):
		print("model declartion - model type : {}\n".format(self.information_center.model_type))
		if self.information_center.model_type == 'BiLSTM':
			model = BiLSTM(self.information_center)
		elif self.information_center.model_type in cust_models:
			model = cust(self.information_center)
		elif self.information_center.model_type in basis_cust_models:
			model = basis_cust(self.information_center)
		else:
			print(" model type \"{}\" =====> UNEXPECTED MODEL TYPE !! ".format(self.information_center.model_type))
			exit()
		if torch.cuda.is_available(): model.cuda()
		return model

	def train(self):
		train_data = self.information_center.train_data
		num_train = len(train_data)
		BATCH_SIZE = self.information_center.BATCH_SIZE
		EPOCH = self.information_center.EPOCH
		VALID_STEP = self.information_center.VALID_STEP
		num_update = 0
		for epoch in range(1, EPOCH+1):
			# DATA PREPARATION : SHUFFLING
			shuffled_indices = np.random.permutation(np.arange(num_train))
			data = train_data[shuffled_indices].copy()
			
			for i in tqdm(range(0, num_train, BATCH_SIZE)):
				# GRADIENT INITIALIZATION
				self.model.zero_grad()

				# DATA PREPARATION : MINI BATCH
				batch = data[i:i+BATCH_SIZE]
				batch_size  	= len(batch)
				
				# DATA PREPARTION : SORT BATCH
				(x_batch, x_lens_batch, 
				usr_batch, prd_batch,
				y_batch) = self.information_center.split_data(batch, sorting=True)
				
				# FORWARD
				if self.information_center.model_type == 'BiLSTM':
					predict_batch = self.model(x_batch, x_lens_batch)
				else: # cust, basis_cust using meta information
					predict_batch = self.model(x_batch, x_lens_batch, usr_batch, prd_batch)
				
				# LOSS
				loss = self.criterion(predict_batch, self.information_center.to_var(torch.from_numpy(y_batch.astype(np.int64))))
				loss.backward()
				# GRADIENT CLIPPING
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
				# GRADIENT UPDATE
				self.optimizer.step()

				# History Trace
				self.logger.train_loss.accumulate_loss(loss=loss,num_data=batch_size)
				num_update += 1
				# VALIDATION
				if num_update%VALID_STEP==0:
					# EVALUATION
					dev_loss, dev_acc, dev_rmse = self.evaluation(self.information_center.dev_data)
					
					# History Trace
					self.logger.train_loss.log()
					self.logger.dev_loss.log(performance=dev_loss)
					self.logger.dev_rmse.log(performance=dev_rmse)
					update = self.logger.dev_acc.log(performance=dev_acc)
					
					if update:
						torch.save(self.model.state_dict(), self.param_dir)

					# SHOW CURRENT PERFORMANCE
					tqdm.write("""		
	Model Type : {}
	EPOCH {} =====> TRAIN LOSS : {:.4f}
	VALIDATION LOSS     : {:2.4f}    
	VALIDATION ACCURACY : {:2.2f}%    =====> BEST {:2.2f}%
	VALIDATION RMSE     : {:2.4f}""".format(
					self.information_center.model_type,
					epoch, self.logger.train_loss.history[-1], 
					dev_loss,
					dev_acc*100, self.logger.dev_acc.best*100,
					dev_rmse,
					))				
		return

	def evaluation(self, dataset):
		tqdm.write("	EVALUATION ... ")
		BATCH_SIZE = self.information_center.BATCH_SIZE
		# HISTORY DECLARATION
		num_data = len(dataset)
		loss_trace = 0
		predicted_label = np.empty(num_data).astype(np.int64)
		target_label = np.empty(num_data).astype(np.int64)
		for i in range(0, num_data, BATCH_SIZE):
			# DATA PREPARATION : MINI BATCH
			batch = dataset[i:i+BATCH_SIZE]
			batch_size  	= len(batch)
			
			# DATA PREPARTION : SORT BATCH
			(x_batch, x_lens_batch, 
			usr_batch, prd_batch,
			y_batch) = self.information_center.split_data(batch, sorting=True)
			
			# FORWARD
			if self.information_center.model_type == 'BiLSTM':
				predict_batch = self.model(x_batch, x_lens_batch)
			else: # cust, basis_cust using meta information
				predict_batch = self.model(x_batch, x_lens_batch, usr_batch, prd_batch)
			
			# LOSS
			loss = self.criterion(predict_batch, self.information_center.to_var(torch.from_numpy(y_batch.astype(np.int64))))
			loss_trace += (loss.cpu().data.numpy() * batch_size)

			# ACCURACY, RMSE
			_, prediction = torch.max(predict_batch, dim=1)
			prediction = prediction.cpu().data.numpy()
			predicted_label[i:i+batch_size] = prediction
			target_label[i:i+batch_size] = y_batch
		
		loss = loss_trace / num_data
		acc = (predicted_label==target_label).mean()
		rmse = ((predicted_label-target_label)**2).mean()**0.5
		return loss, acc, rmse

	def test(self):
		# LOAD PRETRAINED PARAMETERS
		state_dict = torch.load(self.param_dir)
		own_state = self.model.state_dict()
		for name, param in state_dict.items(): own_state[name].copy_(param)

		# EVALUATION
		_, test_acc, test_rmse = self.evaluation(self.information_center.test_data)

		# SAVE FINAL EVALUATION PERFORMANCE
		performace = """
	" Evaluation with test data set "
	<< Model Type : {} >> 
	TEST ACCURACY : {:2.2f}%
	TEST RMSE     : {:2.4f}

	DEV  ACCURACY : {:2.2f}%
	DEV  RMSE     : {:2.4f} 
	""".format(
			self.information_center.model_type,
			test_acc*100, 
			test_rmse,
			self.logger.dev_acc.best*100,
			self.logger.dev_rmse.history[self.logger.dev_acc.best_idx],
			)
		print(performace)
		return

classifier = modelClassifier()
classifier.train()
classifier.test()
