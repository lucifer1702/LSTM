from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
df = pd.read_csv('train/train.csv')
print(df.head())
df.dropna()
x = df.drop('label', axis=1)
y = df['label']
voc_Size = 5000
msg = x.copy()
msg.reset_index(inplace=True)
print(msg['title'])
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []  # list
for i in range(0, len(msg)):
    print(i)
    revie = re.sub('[^a-zA-Z]', '  ', msg['title'][i])
    revie = revie.lower()
    revie = revie.split()
    revie = [ps.stem(word)
             for word in revie if not word in stopwords.words('english')]
    revie = ' '.join(revie)
    corpus.append(revie)
one = [one_hot(words, voc_Size)for words in corpus]
print(one)
sent = 20
emd = pad_sequences(one, padding='pre', maxlen=sent)
print(emd)
embedd_feaures = 40
model = Sequential()
model.add(Embedding(voc_Size, embedd_feaures, input_length=sent))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
x_final = np.array(emd)
y_final = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(
    x_final, y_final, test_size=0.3, random_state=42)
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64)
