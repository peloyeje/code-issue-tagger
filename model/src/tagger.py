import torch
import torch.nn as nn
import numpy as np
from preprocessing.utils import *
from embedder import Embedder
from models.convolution import *

class Tagger:

	def __init__(self, string, trained_model_PATH, vocabulary = './XX/vocabulary.pkl', tag = './XX/tags.pkl', status = 'raw'):
		self.string = string
		self.trained_model_PATH = trained_model_PATH
		self.status = status
		self.model =  None
		self.clean_string = None
		self.output = None
		self.embedding = Embedder(vocabulary, tag)

	def _initialize(self):
		if self.model is None :
			#self.model = nn.Module()
			self.model = ConvModel(embedding_dim=32, vocab_size=len(self.embedding._vocabulary), seq_len=250)
	@property
	def _initialized(self):
		return self.model is not None

	@property
	def _validstring(self):
		return isinstance(self.string, str)
	
	def clean(self):
		if self._validstring:
			try:
				self.clean_string =  unicodeToAscii(self.string)
				self.status = 'clean'
			except:
				raise ValueError('Not a string')

	def preprocess(self):
		if self.status == 'clean':
			self.clean_string = preprocess_string(self.clean_string)
			self.status = 'preprocessed'
		else: 
			self.clean()
			self.clean_string = preprocess_string(self.clean_string)
			self.status = 'preprocessed'

	def embed(self):
		if self.status == 'preprocessed':
			self.embedded = self.embedding.embed(self.clean_string)
			self.embedded = np.asarray(self.embedded, dtype = int)
			self.status = 'embedded'

	def loadmodel(self):
		#try:
		self._initialize()
		#except:
		#	print('could not initialize model')

		if self._initialized and isinstance(self.model, nn.Module):
			#self.model = torch.load(self.trained_model_PATH, map_location = 'cpu')
			self.model.load_state_dict(torch.load(self.trained_model_PATH, map_location='cpu'))
			self.model.eval()
		else:
			self.model = None
			print('could not load model')
	
	def decrypt_top_tags(self, n=10):
		if self.output is not None:
			output_ids = np.argpartition(self.output, -n)[-n:]
			top_tags = np.array(self.embedding._tags)[output_ids]
			top_prob = self.output[output_ids]
		
			return dict(zip(top_tags,top_prob))

		else:
			print('no inference outputs')

	def predict(self):
		#try:
		self.clean()
		#except: 
		#	print('data cleaning error')
		#try:
		self.preprocess()
		#except: 
		#	print('data preprocessing error')
		#try:
		self.embed()
		#except: 
		#	print('data embedding error')
		#try:
		self.loadmodel()

		if self.status == 'embedded' and self.model is not None:
			self.output = self.model(torch.from_numpy(self.embedded))
