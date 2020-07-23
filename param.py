USE_PRETRAINED_EMBEDDING = True # 是否要使用 pre-trained word embedding
TRAIN_TEST_PORTION = 0.8 # 訓練與測試集的切分比例
NEGATIVE_SAMPLE_MULTIPLE = 3 # negative samples 的相較於 positive samples 的倍數，設為 2 的話代表訓練時 positive : negative 的比例是 1 : 2
USE_WORD_DATA = True # 是否要使用經過分詞的資料集，若為 Ture 則使用經過CRF分詞的資料；若為 False 則不經分詞，直接以單一漢字作為輸入
EMBEDDING_DIMENSION = 100 # Embedding維度
SENTENCE_MAX_LEN = 1000 #設定一句的最大長度
BATCH_SIZE = 32 # 設定每次進行梯度下降計算的資料筆數
EPOCHS = 30 # 設定訓練的總體迭代次數
USE_EARLY_STOPPING = False # 是否要使用 Early Stopping，意即提早停止訓練避免 Overfitting
EARLY_STOPPING_PATIENCE = 3 # Early Stopping 的耐心值，意即若超過該數量個 epoch，validtion dataset 的表現仍無提升，即停止訓練

LSTM_UNITS = 128 # LSTM隱藏層單元數

TAGS = [0, 1]

CHAR_BASED_WORD2IDX_FILE = "embedding/character/word2idx.pkl"
CHAR_BASED_EMBEDDING_FILE = "embedding/character/word2vec.taisho.dim100.model"
WORD_BASED_WORD2IDX_FILE = "embedding/word/word2idx.pkl"
WORD_BASED_EMBEDDING_FILE = "embedding/word/fasttext.taisho.dim100.model"
OUTPUT_MODEL_FILE = "out/dhd_model.h5"

