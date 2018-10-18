#!/usr/bin/env python3

import collections
import pathlib
import pickle
import sys

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import utils

# Variables
VOCABULARY_SIZE = 10000
TAG_SIZE = 250

OUTPUT_PATH = pathlib.Path(__file__).parent.joinpath('../../output/').resolve()
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    """
    Builds vocabulary list and tag list from input dataset
    """

    dataset = pathlib.Path(sys.argv[1])

    vocabulary = collections.Counter()
    tags = collections.Counter()

    print(f'Opening {dataset} ...')
    df = pd.read_csv(dataset).dropna()

    print(f'Counting words and tags ...')
    words = (w for s in df['body'].astype(str) for w in s.split())
    df['tags'] = df['tags'].str.split("|", expand=False)
    vocabulary.update(words)
    tags.update((t for l in df['tags'] for t in l))

    vocabulary = {
        w: i for (w, _), i in zip(vocabulary.most_common(VOCABULARY_SIZE), range(VOCABULARY_SIZE))
    }
    tags = {
        w: i for (w, _), i in zip(tags.most_common(TAG_SIZE), range(TAG_SIZE))
    }

    # Saving data
    with (OUTPUT_PATH / 'vocabulary.pkl').open('wb') as f:
        print(f'Saving vocabulary to {f.name} ...')
        pickle.dump(file=f, obj=vocabulary)
    with (OUTPUT_PATH / 'tags.pkl').open('wb') as f:
        print(f'Saving tags to {f.name} ...')
        pickle.dump(file=f, obj=tags)
