#!/usr/bin/env python3

import pathlib

import pandas as pd
import torch
from torch.utils import data

from .embedder import Embedder

class Dataset(data.Dataset):
    def __init__(self, vocabulary, tags, dataset):
        files = {
            '_' + k : pathlib.Path(p) for k, p in locals().items()
            if k is not 'self'
        }

        if any(not f.is_file() for f in files.values()):
            raise ValueError('One of the specified files is missing')
        else:
            self.__dict__.update(files)

        self._dataset = self._open_dataset(self._dataset)
        self._embedder = Embedder(vocabulary=self._vocabulary,
                                  tags=self._tags)

    def set_options(self, max_length=250):
        self.max_length = max_length

    def _open_dataset(self, path):
        try:
            return pd.read_csv(path, index_col=False).dropna()
        except Exception as e:
            raise e

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        X, y = self._dataset.iloc[index, [1, 2]]
        return self._embedder.embed(X, y, pad=True, pad_length=self.max_length)
