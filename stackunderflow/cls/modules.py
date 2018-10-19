import pathlib
import pickle
import numpy as np
import re
import string
import unicodedata
import nltk

CONTENT_REGEX = re.compile(r"<p>(.*)</p>")
TAG_REGEX = re.compile(r"(<.*>)")
WHITESPACE_REGEX = re.compile(r"\s+")
TO_REPLACE = string.punctuation.replace("#","")
MATRIX = str.maketrans(TO_REPLACE + string.digits, " " * len(TO_REPLACE + string.digits))
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
ALL_LETTERS = string.ascii_letters + string.whitespace

def extract_text_content(text):
    """
    Returns text content of p tags and remove other HTML tags from string
    """
    text = " ".join(CONTENT_REGEX.findall(text))
    text = TAG_REGEX.sub("", text)
    return text

def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def preprocess_string(text):
    """
    Filter string : removes tokens, stopwords, and lowercase everything
    """
    text = unicodeToAscii(text)
    text = text.lower()
    text = text.translate(MATRIX)
    text = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(text)




class Embedder():
    def __init__(self, vocabulary, tags):
        files = {
            '_' + k : pathlib.Path(p) for k, p in locals().items()
            if k is not 'self'
        }

        if any(not f.is_file() for f in files.values()):
            raise ValueError('One of the specified files is missing')
        else:
            self.__dict__.update(files)

        self._vocabulary = self._open_pickle(self._vocabulary)
        self._tags = self._open_pickle(self._tags)

    def _open_pickle(self, path):
        try:
            with path.open('rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise e

    def _get_word_index(self, word):
        try:
            return self._vocabulary[word]
        except KeyError:
            return None

    def _get_tag_index(self, tag):
        try:
            return self._tags[tag]
        except KeyError:
            return None

    def _pad(self, array, max_size):
        """
        Front pads an array of size max_size
        """
        output = np.zeros(max_size)
        if array.size != 0:
            trunc = array[:max_size]
            output[-len(trunc):] = trunc
        return output

    def embed(self, text, tags=None, pad=True, pad_length=250):
        """
        Parameters
        ----------
        text: str
            Cleaned post body
        tags: list, optional
            List of tags associated to the post
        pad: bool
            Should we pad the sequence ?
        pad_length: int
            Length of the final vector
        """

        indexes = (self._get_word_index(w) for w in text.split())
        X = np.array([i for i in indexes if i])
        if pad:
            X = self._pad(X, max_size=pad_length)

        if not tags:
            return X
        else:
            indexes = (self._get_tag_index(t) for t in tags)
            y = np.zeros(len(self._tags))
            y[[i for i in indexes if i]] = 1
            return X, y
        
import torch
import torch.nn.functional as F

class ScaledEmbedding(torch.nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)



class ConvModel(torch.nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 vocab_size=10000,
                 seq_len=250,
                 target_size=100):
        super(ConvModel,self).__init__()

        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._seq_len = seq_len
        self._target_size = target_size
        self._max_pool_kernel = 2

        self.embeddings = ScaledEmbedding(self._vocab_size, self._embedding_dim)
        self.conv1 = torch.nn.Conv1d(self._embedding_dim, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 32, 5, padding=2)
        self.mp1 = torch.nn.MaxPool1d(self._max_pool_kernel)
        self.mp2 = torch.nn.MaxPool1d(self._max_pool_kernel)
        self.fc1 = torch.nn.Linear(62 * 32, 1024)
        self.fc2 = torch.nn.Linear(1024, self._target_size)

    def forward(self, x):
        x = self.embeddings(x).permute(0,2,1)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.mp1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.mp2(x)
        x = x.reshape(-1, 62 * 32)
        x = self.fc1(x)
        x = F.dropout(x, 0.2)
        x = self.fc2(x)
        return x