import torch.nn as nn
from utils import *
from embedder import embedder 

class tagger:

	def __init__(self, string, trained_model, clean_string = None, status = 'raw', model = None):
		self.string = string
		self.model =  None
		self.trained_model = trained_model
		self.clean_string = clean_string
		self.status = status

	def _initialize(self):
		if self.model is None :
			self.model = nn.Module()

	@property
	def _initialized(self):
		return self._model is not None

	@property
	def _validstring(self):
		return isinstance(self.string, str):
	
	def clean(self):
		if self._validstring:
			self.clean_string =  extract_text_content(self.string)
			self.status = 'clean'

	def prepropress(self):
		if self.status == 'clean':
			self.clean_string = preprocess_string(self.clean_string)
			self.status = 'preprocessed'
		else: 
			self.clean()
			self.clean_string = preprocess_string(self.clean_string)
			self.status = 'preprocessed'

	def embed(self)
		if self.status == 'preprocessed':
			self.embedded = embedd(PATH, self.clean_string)
			self.status = 'embedded'

	def loadmodel(self):
	try:
		self._initialize()
	except:
		print('could not initialize model')
	
	if self._initialized and isinstance(trained_model, nn.Module):
		self.model._load_state_dict(trained_model)
		self.model.eval()
	else:
		self.model = None
		print('could not load model')

	def predict(self):
		if self.status = 'embedded' and self.model is not None:
			return self.model(self.embedded)
