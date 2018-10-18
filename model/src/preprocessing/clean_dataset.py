#!/usr/bin/env python3

import pathlib
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

import utils

# Variables
VIEW_COUNT_THRESHOLD = 100
COLUMNS_TO_KEEP = ['body', 'tags']

OUTPUT_PATH = pathlib.Path(__file__).parent.joinpath('../../output/processed/').resolve()
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    """
    The purpose of this script is to clean Stack Overflow posts dump.
    """

    source_path = pathlib.Path(sys.argv[1])
    datasets = list(source_path.glob('full-*.csv.gz'))[:5]

    for i, dataset in enumerate(datasets):
        print(f'{i+1}/{len(datasets)} - Opening {dataset} ...')

        df = pd.read_csv(dataset).set_index('id')

        # Dropping least viewed questions
        df = df[df['view_count'] > VIEW_COUNT_THRESHOLD]

        # Extract raw text from SO HTML markup
        df['body'] = df['body'].progress_apply(utils.extract_text_content)
        # Preprocess raw text
        df['body'] = df['body'].progress_apply(utils.preprocess_string)
        # Remove empty strings
        df['body'] = df['body'].replace(r'^\s*$', value=np.nan, regex=True)
        df = df.dropna()

        print(f'Saving processed dataset to {OUTPUT_PATH / dataset.name} ...')
        df[COLUMNS_TO_KEEP].to_csv(OUTPUT_PATH / dataset.name,
                                   compression='gzip',
                                   index=False)