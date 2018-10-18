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
TAG_SIZE = 1000

OUTPUT_PATH = pathlib.Path(__file__).parent.joinpath('../../output/').resolve()
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    """
    The purpose of this script is to build training and test sets
    from Stack Overflow posts dump.
    """

    source_path = pathlib.Path(sys.argv[1])
    datasets = list(source_path.glob('full-*.csv.gz'))[:3]

    vocabulary = collections.Counter()
    tags = collections.Counter()

    for i, dataset in enumerate(datasets):
        print(f'{i+1}/{len(datasets)} - Opening {dataset} ...')

        df = pd.read_csv(dataset)

        # For safety checks
        df = df.dropna()

        # Computing vocabulary
        words = (w for s in df['body'].astype(str) for w in s.split())
        vocabulary.update(words)

        # Computing tags
        df['tags'] = df['tags'].str.split("|", expand=False)
        tags.update((t for l in df['tags'] for t in l))

    vocabulary = [w for w, c in vocabulary.most_common(VOCABULARY_SIZE)]
    tags = [w for w, c in tags.most_common(TAG_SIZE)]

    # Saving data
    with (OUTPUT_PATH / 'vocabulary.pkl').open('wb') as f:
        pickle.dump(file=f, obj=vocabulary)
    with (OUTPUT_PATH / 'tags.pkl').open('wb') as f:
        pickle.dump(file=f, obj=tags)
