import pandas as pd
import numpy as np
df=pd.read_csv('train/train.csv')
print(df.head())
df.dropna()
x=df.drop('label',axis=1)
y=df['label']
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
voc_Size=5000
msg=x.copy()
msg.reset_index(inplace=True)
print(msg['title'])