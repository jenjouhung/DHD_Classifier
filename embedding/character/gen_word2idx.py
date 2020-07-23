#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models

MODEL_FILE = 'word2vec.taisho.dim100.model'

def main():
    model = models.Word2Vec.load(MODEL_FILE)

    word2idx = { w: i for i, w in enumerate(model.wv.index2word) }
    word2idx['UNK'] = len(word2idx)
    word2idx['ENDPAD'] = len(word2idx)
    import pickle
    with open("word2idx.pkl", 'wb') as wordf:
        pickle.dump(word2idx, wordf)

if __name__ == "__main__":
    main()


