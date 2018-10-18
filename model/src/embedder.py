import pathlib
import pickle

import numpy as np

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
            return self._vocabulary.index(word)
        except:
            return None

    def _get_tag_index(self, tag):
        try:
            return self._tags.index(tag)
        except:
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
