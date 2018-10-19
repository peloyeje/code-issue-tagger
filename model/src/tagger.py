import torch
import torch.nn as nn
import numpy as np
from preprocessing.utils import *
from embedder import Embedder
from models.convolution import *

class Model:
	def __init__(self, trained_model_PATH, vocabulary = './XX/vocabulary.pkl', tag = './XX/tags.pkl', status = 'raw'):
		self.trained_model_PATH = trained_model_PATH
		self.vocabulary = vocabulary
		self.tag =tag
		self.model =  None
		self.embedding = Embedder(vocabulary, tag)
		self.loadmodel()

	def _initialize(self):
		if self.model is None :
			self.model = ConvModel(embedding_dim=32, vocab_size=len(self.embedding._vocabulary), seq_len=250)

	def loadmodel(self):
		
		self._initialize()

		if self._initialized and isinstance(self.model, nn.Module):
			self.model.load_state_dict(torch.load(self.trained_model_PATH, map_location='cpu'))
			self.model.eval()
		else:
			self.model = None
			print('could not load model')

	@property
	def _initialized(self):
		return self.model is not None

class Tagger:

	def __init__(self, string, inp_model):
		self.string = string
		self.status = 'raw'
		self.clean_string = None
		self.output = None
		self.embedding = inp_model.embedding
		self.model = inp_model.model


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

	
	def decrypt_top_tags(self, n=10):
		if self.output is not None:
			out = self.output.detach().numpy()[0]
			output_ids = np.argpartition(out, -n)[-n:]
			top_tags = [list(self.embedding._tags.keys())[i] for i in output_ids]
			top_prob = out[output_ids]
		
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
		if self.status == 'embedded':
			self.output = self.model(torch.from_numpy(self.embedded).view(1,250))
