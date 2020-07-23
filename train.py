# -*- coding: utf-8 -*- 
import numpy as np
import pickle
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import dataload
import param
from model import birnn
#from model import att_birnn

WORD2IDX_FILE = param.WORD_BASED_WORD2IDX_FILE if param.USE_WORD_DATA else param.CHAR_BASED_WORD2IDX_FILE
EMBEDDING_FILE = param.WORD_BASED_EMBEDDING_FILE if param.USE_WORD_DATA else param.CHAR_BASED_EMBEDDING_FILE

#載入訓練與測試資料集
train_sentences, train_tags, test_sentences, test_tags = dataload.load()

# 載入詞彙與編號的對應表 word2idx，該表來自 CBETA word embedding 的 vocabulary
wordf = open(WORD2IDX_FILE, 'rb')
word2idx = pickle.load(wordf)
n_words = len(word2idx) #詞彙總數

tags = param.TAGS
n_tags = len(tags) #標籤類型總數：1 和 0 兩種

# 產生Deep Learning用的訓練與測試資料矩陣
# 填充每一段文字到固定的長度長度，採用Keras提供的 pad_sequences 函式進行填充
from keras.preprocessing.sequence import pad_sequences
X_train = [[word2idx[w] if w in word2idx else word2idx['UNK'] for w in s] for s in train_sentences] # 這裡順道處理如果遇到沒看過的詞的時候，會指定為 UNK 所對應的編號 (代表未知詞)
X_train = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=X_train, padding="post",value=n_words - 1)

y_train = np.array(train_tags)

# 打亂訓練資料的順序，因為原本的資料前半部分是positive資料，後半部分是negative的，若不先打亂，在訓練的時候進行validation set分割時會變成全部都是某一種單一類型的資料，會影響訓練
train_index = np.arange(len(y_train))
np.random.shuffle(train_index)
X_train = X_train[train_index,:]
y_train = y_train[train_index]

# 產生測試資料矩陣
X_test = [[word2idx[w] if w in word2idx else word2idx['UNK'] for w in s] for s in test_sentences]
X_test = pad_sequences(maxlen=param.SENTENCE_MAX_LEN, sequences=X_test, padding="post",value=n_words - 1)

y_test = np.array(test_tags)

embedding_matrix = None
# 如果要使用預訓練的詞向量，載入pre-trained word2vec/fasttext
if param.USE_PRETRAINED_EMBEDDING:
    from gensim import models
    w2v_model = None
    if param.USE_WORD_DATA: 
        w2v_model = models.FastText.load(EMBEDDING_FILE)
    else:
        w2v_model = models.Word2Vec.load(EMBEDDING_FILE)
    embedding_matrix = np.zeros((n_words, param.EMBEDDING_DIMENSION))
    for w, i in word2idx.items():
        if w in w2v_model:
            embedding_vector = w2v_model[w]
            embedding_matrix[i] = embedding_vector
    # 將未知詞給定一個隨機的向量
    unk_index = word2idx['UNK']
    embedding_matrix[unk_index] = np.random.rand(param.EMBEDDING_DIMENSION)

# 建立類神經網路模型
model = birnn.create_model(n_words, n_tags, param.SENTENCE_MAX_LEN, param.LSTM_UNITS, param.EMBEDDING_DIMENSION, embedding_matrix)
#model = att_birnn.create_model(n_words, n_tags, param.SENTENCE_MAX_LEN, param.LSTM_UNITS, param.EMBEDDING_DIMENSION, embedding_matrix)

# 印出模型各層概況
model.summary()

# 設定 Early Stopping
callback_funcs = []
early_stopping = EarlyStopping(monitor='val_acc', patience=param.EARLY_STOPPING_PATIENCE, mode='max')
if param.USE_EARLY_STOPPING:
    callback_funcs.append(early_stopping)

# 開始訓練
history = model.fit(X_train, y_train, batch_size=param.BATCH_SIZE, epochs=param.EPOCHS, validation_split=0.2, verbose=1, callbacks=callback_funcs)

# 將訓練好的模型存檔
model.save(param.OUTPUT_MODEL_FILE)

# 用測試資料集進行測試，估測accuracy
p = model.evaluate(X_test, y_test, verbose=1)
for i in range(len(model.metrics_names)):
    print("{0}:{1}".format(model.metrics_names[i], p[i]))
