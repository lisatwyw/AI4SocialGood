from pathlib import Path
import json, os
import pandas as pd

import os, sys, pickle, json
import pandas as pd
import polars as pol
import numpy as np
from pathlib import Path

#from IPython.display import clear_output;  clear_output()

import textwrap

import nltk
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


import multiprocessing as mp
import re

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio; pio.renderers.default = 'colab' # enable renderer
'''
['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
  'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
  'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
  'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
  'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
'''
from tqdm import tqdm

from PIL import Image
from functools import wraps

from sklearn.manifold import TSNE
#from transformers import AutoTokenizer, FeatureExtractionPipeline, pipeline

import random, math
import torch

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


folder = '/kaggle/input/neiss-2023/'

def get_data():
    with Path( folder + "variable_mapping.json").open("r") as f:
        mapping = json.load(f, parse_int=True)

    for c in mapping.keys():
        mapping[c] = {int(k): v for k, v in mapping[c].items()}

    df = pd.read_csv(folder+"primary_data.csv", parse_dates=['treatment_date'], 
                   dtype={"body_part_2": "Int64", "diagnosis_2": "Int64", 'other_diagnosis_2': 'string'} )

    df2 = pd.read_csv(folder+"supplementary_data.csv",  parse_dates=['treatment_date'], 
                    dtype={"body_part_2": "Int64", "diagnosis_2": "Int64", 'other_diagnosis_2': 'string' } )

    org_columns = df2.columns

    df2['month'] = df2.treatment_date.dt.month
    df2['year'] = df2.treatment_date.dt.year

    df2['severity'] = df2['disposition'].replace(
        {'1 - TREATED/EXAMINED AND RELEASED': 3,
         '2 - TREATED AND TRANSFERRED': 4,
         '4 - TREATED AND ADMITTED/HOSPITALIZED': 5,  # question, more or less severe?
         '5 - HELD FOR OBSERVATION': 2,
         '6 - LEFT WITHOUT BEING SEEN': 1
        })

    df2['age_cate']= pd.cut(
    df2.age,
    bins=[0,65,75,85,95,150],
    labels=["1: 65 or under", "2: 65-74", "3: 74-85", "4: 85-94", "5: 95+"],
    )
    # drop the 3 cases of unknown sex and then map codes to English words
    df2 = df2[df2.sex != 0]

decoded_df2 = df2.copy()
for col in mapping.keys():
    if col != 'disposition':
        decoded_df2[col] = decoded_df2[col].map(mapping[col])    

decoded_df = df.copy()
for col in mapping.keys():
    if col != 'disposition':
        decoded_df[col] = decoded_df[col].map(mapping[col])        

return decoded_df, decoded_df2

if ( 'org_columns' in globals())==False:
  decoded_df, decoded_df2 = get_data()


tst_case_nums = np.setdiff1d( decoded_df2.cpsc_case_number, df.cpsc_case_number )
trn_case_nums = np.setdiff1d( decoded_df2.cpsc_case_number, tst_case_nums )

print( 'If models were to be developed, we may split into trn set of', len(trn_case_nums), 'samples and tst set of',
    len(tst_case_nums), 'samples. \n\tOverlapped indices?', np.intersect1d( tst_case_nums, trn_case_nums ), '\n\n' )
 
