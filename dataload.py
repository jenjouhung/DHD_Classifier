import dataload_word
import dataload_char
import param

def load():
    return dataload_word.load() if param.USE_WORD_DATA else dataload_char.load()
