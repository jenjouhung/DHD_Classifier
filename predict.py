import pickle
import numpy as np
import keras
import param
from wordcut import crfsegmenter

# 標籤種類，需與訓練時的一致
tags = param.TAGS
n_tags = len(tags)

# 載入之前訓練時的 word2idx
WORD2IDX_FILE = param.WORD_BASED_WORD2IDX_FILE if param.USE_WORD_DATA else param.CHAR_BASED_WORD2IDX_FILE
wordf = open(WORD2IDX_FILE, 'rb')
word2idx = pickle.load(wordf)
n_words = len(word2idx)

# 建立雙向 LSTM 模型
#model = birnn.create_model(n_words, n_tags, param.SENTENCE_MAX_LEN, param.LSTM_UNITS, param.EMBEDDING_DIMENSION)
#model.load_weights(param.OUTPUT_MODEL_FILE)
model = keras.models.load_model(param.OUTPUT_MODEL_FILE)

# 請使用者輸入想要進行 NER 標注的句子
sentence = input("請輸入句子：")
cut_sentence = crfsegmenter.cut(sentence)

# 將使用者輸入的句子進行填充到固定的長度
from keras.preprocessing.sequence import pad_sequences
X_test = [[word2idx[w] if w in word2idx else word2idx['UNK'] for w in cut_sentence]]
X_test = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=X_test, padding="post", value=n_words - 1)

# 利用模型進行預測
p = model.predict(np.array([X_test[0]]), verbose=0)
print(p[0][0])
