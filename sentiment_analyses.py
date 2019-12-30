import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pymongo

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

mng_client = pymongo.MongoClient('localhost', 27017)
mng_db = mng_client['assignment2']
collection_name = 'balanced_reviews'
db_cm = mng_db[collection_name].find()

# Expand the cursor and construct the DataFrame
df = pd.DataFrame(list(db_cm))
del df['_id']

import seaborn as sns

sns.countplot(x='positive', data=df)