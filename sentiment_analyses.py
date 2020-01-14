import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pymongo
import dask.dataframe as ddf

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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

mng_client = pymongo.MongoClient('localhost', 27017)
mng_db = mng_client['assignment2']
collection_name = 'balanced_reviews'
db_cm = mng_db[collection_name].find()

# Expand the cursor and construct the DataFrame
reviews = pd.DataFrame(list(db_cm))
del reviews['_id']

import seaborn as sns

# this function is used for apply
def proces_text(row):
    
    the_text = row['review']

    row['review'] = proces_text_only(the_text)    
    
    return row

# this function is used for only text
def proces_text_only(text_only):
    # Remove all the special characters (pro_fea = processed feature)
    pro_fea = re.sub(r'\W', ' ', text_only)
    # remove all single characters
    pro_fea = re.sub(r'\s+[a-zA-Z]\s+', ' ', pro_fea)
    # Remove single characters from the start
    pro_fea = re.sub(r'\^[a-zA-Z]\s+', ' ', pro_fea) 
    # Substituting multiple spaces with single space
    pro_fea = re.sub(r'\s+', ' ', pro_fea, flags=re.I)
    # Removing prefixed 'b'
    pro_fea = re.sub(r'^b\s+', '', pro_fea)
    # Converting to Lowercase
    return pro_fea.lower() 

ddf_reviews = ddf.from_pandas(reviews, npartitions=7)
ddf_rev_pr = ddf_reviews.apply(proces_text, axis=1, meta={'review': 'object', 'positive': 'int64'})

df_rev = ddf_rev_pr.compute()

REV_LIMIT = 100000
rev_negatives = df_rev[df_rev.positive == 0][:REV_LIMIT]
rev_positives = df_rev[df_rev.positive == 1][:REV_LIMIT]

rev_balanced = pd.concat([rev_negatives, rev_positives]).reset_index(drop=True)

X = np.array(list(rev_balanced.loc[:, 'review']))
y = np.array(list(rev_balanced.loc[:, 'positive']))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)

# Create word-to-index dictionary.
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('D:/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# create model
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

#train model
history = model.fit(X_train, y_train, batch_size=128, epochs=7, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:!", score[1])


# function to test custom reviews
def custom_review(review):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(review)

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    
    filtered_sentence = [] 
    
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)

    the_rev = tokenizer.texts_to_sequences(filtered_sentence)

    ent_list = []

    for sublist in the_rev:
        for item in sublist:
            ent_list.append(item)

    instance = pad_sequences([ent_list], padding='post', maxlen=maxlen)

    print('This review is positive') if model.predict(instance)[0][0] > 0.5 else print('This review is negative')
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

custom_review("Wow, this hotel was so amazing. It had a beautiful view and a great breakfast.")
custom_review("This was terrible. What a bad hotel. Breakfast was awful")
custom_review("The hotel was alright. The breakfast was decent but the view was not very nice. However the shower was great")
