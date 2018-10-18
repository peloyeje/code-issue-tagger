#!/usr/bin/env python3

import pathlib

import pandas as pd
import torch
from torch.utils import data

from .embedder import Embedder

class Dataset(data.Dataset):
    def __init__(self, vocabulary, tags, dataset, pad=True, pad_length=250):
        files = {
            k : pathlib.Path(p) for k, p in locals().items()
            if k not in ['self', 'pad', 'pad_length']
        }

        if any(not f.is_file() for f in files.values()):
            raise ValueError('One of the specified files is missing')
        else:
            self.__dict__.update(files)

        self.pad = pad
        self.pad_length = pad_length
        self._data = self._open_dataset(dataset, ['body', 'tags'])
        self._embedder = Embedder(vocabulary=vocabulary,
                                  tags=tags)

    def _open_dataset(self, path, columns=None):
        try:
            df = pd.read_csv(path, index_col=0).dropna()
            return df[columns] if columns else df
        except Exception as e:
            raise e

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        X, y = self._data.iloc[index, [0, 1]]
        return self._embedder.embed(X, y, pad=self.pad,
                                          pad_length=self.pad_length)
