# DHD_Classifier

## 需求套件

* numpy == 1.17.2
* tensorflow
* keras
* scikit-learn
* gensim

## 預訓練詞嵌入

由於 GitHub 檔案大小限制 100MB，由 FastText 所訓練的 CBETA 詞嵌入檔案超過限制無法放進 GitHub 專案目錄，請於如下網址下載：

[下載詞嵌入](https://drive.google.com/file/d/1JQwQ5N2BE4YEif-Y6HZdU07MoEDE7sIH/view?usp=sharing)

請將該檔案置放於 embedding/word/ 目錄中，進行解壓縮：

    tar zxvf word_embeddings.tar.gz

## 檔案說明

* param.py : 模型相關參數設定皆位於此
* dataload.py : 載入訓練與測試資料集
 * dataload_char.py : 載入單字資料集
 * dataload_word.py : 載入經過分詞的資料集
* train.py : 訓練步驟實作
* predict.py : 完成訓練後，可輸入句子取得辨識結果

## 目錄說明

* model/ : 類神經網路模型實作 (目前有 雙向LSTM(birnn.py) 與 雙向LSTM+注意力機制(att_birnn.py) 兩種模型)
* preporcess/ : 資料集預處理相關程式 (處理 test_data/ 裡的原始資料)
* wordcut/ : CRF佛典分詞模型 (進行分詞用)
* embedding/ : 預訓練嵌入
 * embedding/character/ : 預訓練單字嵌入 (以 Word2Vec 訓練)
 * embedding/word/ : 預訓練詞嵌入 (經過分詞，以 FastText 訓練)
* out/ : 輸出目錄，訓練出來的模型參數檔會放在這裡

## 進行訓練

    python train.py

## 進行預測

    python predict.py
