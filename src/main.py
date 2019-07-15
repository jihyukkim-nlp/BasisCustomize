# utility packages
import argparse, os, numpy as np, pickle, random
import logging, colorlog
from tqdm import tqdm

# Pytorch packages
import torch, torch.nn as nn, torch.nn.functional as F

# Dataloader
from model_src.CustomDataset.yelp2013 import yelp2013
from model_src.CustomDataset.polmed import polmed
from model_src.CustomDataset.aapr import aapr

# Pytorch.Ignite Packages
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar

# model
from model_src.model import Classifier


logging.disable(logging.DEBUG)
colorlog.basicConfig(
	filename=None,
	level=logging.NOTSET,
	format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)
parser = argparse.ArgumentParser()
baseline_models = ['BiLSTM']
cust_models = ['word_cust', 'encoder_cust', 'attention_cust', 'linear_cust', 'bias_cust']
basis_cust_models = ['word_basis_cust', 'encoder_basis_cust', 'attention_basis_cust', 'linear_basis_cust', 'bias_basis_cust']
model_choices = baseline_models + cust_models + basis_cust_models
parser.add_argument("--random_seed", type=int, default=33)
parser.add_argument("--model_type", choices=model_choices, help="Give model type.")
parser.add_argument("--domain", type=str, choices=['yelp2013', 'polmed', 'aapr'], default="yelp2013")
parser.add_argument("--num_bases", type=int, default=0)
parser.add_argument("--vocab_dir", type=str)
parser.add_argument("--train_datadir", type=str, default="./processed_data/flat_data.p")
parser.add_argument("--dev_datadir", type=str, default="./processed_data/flat_data.p")
parser.add_argument("--test_datadir", type=str, default="./processed_data/flat_data.p")
parser.add_argument("--word_dim", type=int, default=300, help="word vector dimension")
parser.add_argument("--state_size", type=int, default=256, help="BiLSTM hidden dimension")
parser.add_argument("--meta_dim", type=int, default=64, help="meta embedding latent vector dimension")
parser.add_argument("--key_query_size", type=int, default=64, help="key and query dimension for meta context")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--valid_step", type=int, default=1000, help="evaluation step using dev set")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--pretrained_word_em_dir", type=str, default="")
parser.add_argument("--max_grad_norm", type=float, default=3.0)

args = parser.parse_args()
if 'basis' in args.model_type:
	if args.num_bases==0:
		print(" must input number of bases (\"--num_bases\") for basis_cust model type")
		print(" e.g. python main.py word_basis_cust --num_bases 3")
		exit()
# Manual Random Seed
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.backends.cudnn.deterministic=True

class modelClassifier:
	def __init__(self):

		# Ignite engine
		self.engine = None
		self._engine_ready()

		# Dataloader
		domain_dataloader = {
		'yelp2013':yelp2013(args),
		'polmed':polmed(args),
		'aapr':aapr(args),
		}[args.domain]
		self.train_dataloader = domain_dataloader.train_dataloader
		self.dev_dataloader = domain_dataloader.dev_dataloader
		self.test_dataloader = domain_dataloader.test_dataloader

		# MODEL DECLARATION
		self.model = Classifier(args).to(args.device)
		print("<< Model Configuration >>")
		print(self.model)
		print("*"*50)
		with open("./ModelDescription.txt", "w") as f:
			f.write(repr(self.model))

		# OPTIMIZER DECLARATION
		parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optimizer = torch.optim.Adadelta(parameters, lr=1.0, rho=0.9, eps=1e-6)
		self.criterion = {
		"yelp2013":F.cross_entropy,
		"polmed":F.cross_entropy,
		"aapr":F.binary_cross_entropy,
		}[args.domain]

		# PARAM SAVE DIRECTORY
		self.param_dir = './save_param/'
		if not os.path.exists(self.param_dir): os.mkdir(self.param_dir) # ./save_param
		self.param_dir = os.path.join(self.param_dir,args.domain)
		if not os.path.exists(self.param_dir): os.mkdir(self.param_dir) # ./save_param/{domain}
		self.param_dir = os.path.join(self.param_dir, args.model_type)
		if 'basis_cust' in args.model_type: 
			self.param_dir += '({}).pth'.format(args.num_bases)
		else:
			self.param_dir += '.pth'

	def _init_param(self, model):
		colorlog.critical("[Init General Parameter] >> xavier_uniform_")
		for p in model.parameters():
			if p.requires_grad:
				if len(p.shape)>1:
					nn.init.xavier_uniform_(p)
				else:
					nn.init.constant_(p, 0)
		if args.pretrained_word_em_dir:
			colorlog.critical("[Pretrained Word em loaded] from {}".format(args.pretrained_word_em_dir))
			word_em = np.load(args.pretrained_word_em_dir)
			model.word_em_weight.data.copy_(torch.from_numpy(word_em))
	
	def _init_meta_param(self, model):
		colorlog.critical("[Init Meta Parameter] >> uniform_ [-0.01, 0.01]")
		for name, param in model.meta_param_manager.state_dict().items():
			colorlog.info("{} intialized".format(name))
			nn.init.uniform_(param, -0.01, 0.01)

	def _engine_ready(self):
		colorlog.info("[Ignite Engine Ready]")
		self.engine = Engine(self._update)
		ProgressBar().attach(self.engine) # support tqdm progress bar
		self.engine.add_event_handler(Events.STARTED, self._started)
		self.engine.add_event_handler(Events.COMPLETED, self._completed)
		self.engine.add_event_handler(Events.EPOCH_STARTED, self._epoch_started)
		self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._epoch_completed)
		self.engine.add_event_handler(Events.ITERATION_STARTED, self._iteration_started)
		self.engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration_completed)

	def _update(self, engine, sample_batch):
		target, kwinputs = sample_batch
		# Inference
		predict = self.model(**kwinputs)
		# Loss & Update
		loss = self.criterion(input=predict, target=target)
		loss.backward()
		nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
		# Loss logging
		self.engine.state.train_loss += loss.item()/args.valid_step
		return loss.item() # engine.state.output

	def _iteration_started(self, engine):
		self.model.zero_grad()
		self.optimizer.zero_grad()
	def _iteration_completed(self, engine):
		self.optimizer.step()
		# Evaluation
		if self.engine.state.iteration % args.valid_step == 0:
			dev_acc, dev_rmse = self.evaluation(self.dev_dataloader)
			if self.engine.state.best_dev_acc < dev_acc:
				self.engine.state.best_dev_acc = dev_acc
				self.engine.state.dev_rmse = dev_rmse
				torch.save(self.model.state_dict(), self.param_dir)
			colorlog.info("""		
Model Type : {}
EPOCH {} =====> TRAIN LOSS : {:.4f}
VALIDATION ACCURACY : {:2.2f}%    =====> BEST {:2.2f}%
VALIDATION RMSE     : {:2.4f}""".format(
				args.model_type,
				self.engine.state.epoch, self.engine.state.train_loss, 
				dev_acc*100, self.engine.state.best_dev_acc*100,
				dev_rmse,
				))	
			self.engine.state.train_loss = 0
	def _started(self, engine):
		# Model Initialization
		self._init_param(self.model)
		if 'cust' in args.model_type: 
			self._init_meta_param(self.model)
		self.model.train()
		self.model.zero_grad()
		self.optimizer.zero_grad()
		# ignite engine state intialization
		self.engine.state.best_dev_acc = -1
		self.engine.state.dev_rmse = -1
		self.engine.state.train_loss = 0
	def _completed(self, engine):
		colorlog.info("*"*20+" Training is DONE" + "*"*20)
	def _epoch_started(self, engine):
		colorlog.info('>' * 50)
		colorlog.info('EPOCH: {}'.format(self.engine.state.epoch))
	def _epoch_completed(self, engine):
		pass

	def evaluation(self, dataloader):
		colorlog.info("	EVALUATION ... ")
		# HISTORY DECLARATION
		num_data = len(dataloader.dataset)
		predicted_label = np.empty(num_data).astype(np.int64)
		target_label = np.empty(num_data).astype(np.int64)
		self.model.eval()
		with torch.no_grad():
			batch_size = dataloader.batch_size
			for i_batch, sample_batch in enumerate(dataloader):
				target_batch, kwinputs = sample_batch
				predict_batch = self.model(**kwinputs)
				
				# ACCURACY, RMSE
				_, predict_batch = torch.max(predict_batch, dim=1)
				predicted_label[i_batch*batch_size:(i_batch+1)*batch_size] = predict_batch.cpu().data.numpy()
				target_label[i_batch*batch_size:(i_batch+1)*batch_size] = target_batch.cpu().data.numpy()
			
			acc = (predicted_label==target_label).mean()
			rmse = ((predicted_label-target_label)**2).mean()**0.5
		self.model.train()
		return acc, rmse

	def train(self):
		self.engine.run(self.train_dataloader, max_epochs=args.epoch)
	def test(self):
		# LOAD PRETRAINED PARAMETERS
		state_dict = torch.load(self.param_dir)
		self.model.load_state_dict(state_dict)

		# EVALUATION
		test_acc, test_rmse = self.evaluation(self.test_dataloader)

		# SAVE FINAL EVALUATION PERFORMANCE
		colorlog.info("""
	" Evaluation with test data set "
	<< Model Type : {} >> 
	TEST ACCURACY : {:2.2f}%
	TEST RMSE     : {:2.4f}

	DEV  ACCURACY : {:2.2f}%
	DEV  RMSE     : {:2.4f} 
	""".format(
			args.model_type,
			test_acc*100, 
			test_rmse,
			self.engine.state.best_dev_acc*100,
			self.engine.state.dev_rmse,
			))
		return

classifier = modelClassifier()
try:
	classifier.train()
except KeyboardInterrupt:
	print("KeyboardInterrupt occurs")
	print("Start Test Evaluation")

classifier.test()
