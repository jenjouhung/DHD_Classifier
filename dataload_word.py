from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random

import param

POSITIVE_FILE = "test_data/positive_seg.txt"
NEGATIVE_FILE = "test_data/negative_seg.txt"

MAX_WORDS = param.SENTENCE_MAX_LEN

def load():
    positive_sentences = []
    negative_sentences = []
    train_sentences = []
    test_sentences = []

    with open(POSITIVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split(' ')
            positive_sentences.append(text)

    with open(NEGATIVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split(' ')
            negative_sentences.append(text)

    # 將句子順序打亂
    random.shuffle(positive_sentences)
    random.shuffle(negative_sentences)

    positive_tags = [1] * len(positive_sentences)
    negative_tags = [0] * len(negative_sentences)

    # 計算 positive 和 negative 的混合比例
    positive_portion_index = int(len(positive_sentences) * param.TRAIN_TEST_PORTION)
    negative_portion_index = positive_portion_index * param.NEGATIVE_SAMPLE_MULTIPLE

    train_sentences = positive_sentences[:positive_portion_index] + negative_sentences[:negative_portion_index]
    train_sentences = [ x[:MAX_WORDS] if len(x) > MAX_WORDS else x for x in train_sentences ] # 過長的句子進行截斷
    train_tags = positive_tags[:positive_portion_index] + negative_tags[:negative_portion_index]
    test_sentences = positive_sentences[positive_portion_index:] + negative_sentences[negative_portion_index:]
    test_sentences = [ x[:MAX_WORDS] if len(x) > MAX_WORDS else x for x in test_sentences ] # 過長的句子進行截斷
    test_tags = positive_tags[positive_portion_index:] + negative_tags[negative_portion_index:]
    return train_sentences, train_tags, test_sentences, test_tags


if __name__ == '__main__':
    train_data, train_tag, test_data, test_tag = load()
    print(len(train_data))
    print(train_data[5], train_tag[5])
    print(test_data[5], test_data[5])
    print(len(test_data[5]))

