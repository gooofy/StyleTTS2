#!/bin/env python3

import yaml
# import shutil
import sys
# import string
# import pickle

from datasets import load_from_disk, concatenate_datasets
# from dataloader import build_dataloader

from tqdm import tqdm

# from transformers import BertTokenizerFast
# from tokenizers import Tokenizer

LIMIT = 100000
DATA_FOLDER = '../PL-BERT/wikipedia_20220301.de.processed/'
OOD_TXT = 'Data.de/OOD_texts.txt'

# config_path = "Configs/config_de.yml"
# config = yaml.safe_load(open(config_path))

# print (f"load_from_disk({config['data_folder']})...")
# dataset = load_from_disk(config['data_folder'])

print (f"load_from_disk({DATA_FOLDER})...")
dataset = load_from_disk(DATA_FOLDER)


print (f"dataset: {dataset}")
# '25'
print (f"   dataset[0]['id']={dataset[0]['id']}")
# 'https://en.wikipedia.org/wiki/Autism'
print (f"   dataset[0]['url']={dataset[0]['url']}")
# 'Autism'
print (f"   dataset[0]['title']={dataset[0]['title']}")
# [43685, 23, 8, 105463, 5784, 4036, 20, 3561, 19, 853 ...]
print (f"   dataset[0]['input_ids'][:10]={dataset[0]['input_ids'][:10]}")
print (f"   dataset[0]['inputs'][:10]={dataset[0]['inputs'][:10]}")
# ['ˈɔːtɪzəm', 'ɪz', 'ɐ', 'nˌʊɹɹoʊdvˈɛləpmˈɛntəl', 'dɪsˈoːɹdɚ', 'kˈæɹɪktɚɹˌaɪzd', 'bˈaɪ', 'dˈɪfɪkˌʌltiz', 'wɪð', ...
print (f"   dataset[0]['phonemes'][:10]={dataset[0]['phonemes'][:10]}")

# dataset: Dataset({
#    features: ['id', 'url', 'title', 'inputs', 'input_ids', 'phonemes'],
#    num_rows: 2665357
# })
#    dataset[0]['id']=29
#    dataset[0]['url']=https://de.wikipedia.org/wiki/Liste%20von%20Autoren/E
#    dataset[0]['title']=Liste von Autoren/E
#    dataset[0]['input_ids'][:10]=[24159, 3804, 64569, 12, 8084, 16587, 13, 16, 12, 13238]
#    dataset[0]['inputs'][:10]=['ea', 'charles', 'eastman', '(', '1858', '##1939', ')', ',', '(', 'indianer']
#    dataset[0]['phonemes'][:10]=['ea', 'charles', 'eastman', '(', '1858', '1939', ')', ',', '(', 'indianer']

cnt = 0

subset = dataset.shuffle().select(range(LIMIT))

with open (OOD_TXT, 'w') as oodf:

    for data in tqdm(subset):

        txt = ''

        for token in data['inputs']:
            if token[:2] != '##':
                if txt:
                    txt += ' ' + token
                else:
                    txt = token
            else:
                txt += token[2:]

        # print (txt)
        oodf.write (f"{txt}|anything\n")
        #cnt += 1

        #if cnt >= LIMIT:
        #    break

