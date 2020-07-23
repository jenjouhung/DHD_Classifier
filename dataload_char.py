from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 1000

POSITIVE_FILE = "test_data/positive.txt"
NEGATIVE_FILE = "test_data/negative.txt"

def load():
    positive_sentences = []
    positive_tags = []
    negative_sentences = []
    negative_tags = []
    train_sentences = []
    test_sentences = []
    all_sentences = []
    all_tags = []

    with open(POSITIVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            text = list(line.strip())
            positive_sentences.append(text)
            positive_tags.append(1)

    with open(NEGATIVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            text = list(line.strip())
            negative_sentences.append(text)
            negative_tags.append(0)

    train_sentences = positive_sentences[:71] + negative_sentences[:355]
    train_sentences = [ x[:MAX_WORDS] if len(x) > MAX_WORDS else x for x in train_sentences ]
    train_tags = positive_tags[:71] + negative_tags[:355]
    test_sentences = positive_sentences[71:] + negative_sentences[355:400]
    test_sentences = [ x[:MAX_WORDS] if len(x) > MAX_WORDS else x for x in test_sentences ]
    test_tags = positive_tags[71:] + negative_tags[355:400]
    return train_sentences, train_tags, test_sentences, test_tags


if __name__ == '__main__':
    train_data, train_tag, test_data, test_tag = load()
    print(train_data[5], train_tag[5])
    print(test_data[5], test_data[5])
    print(len(test_data[5]))

#tk = Tokenizer(oov_token='UNK')
#tk.fit_on_texts(all_sentences)
#tk.word_index = {e:i for e,i in tk.word_index.items() if i <= num_words}
#tk.word_index[tk.oov_token] = num_words + 1 
#print(tk.word_index)
#X_train = tk.texts_to_sequences(positive_sentences)
#print(len(positive_sentences[1]))
#print(positive_sentences[1])
#print(len(X_train[1]))
#print(X_train[1])
