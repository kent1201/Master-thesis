# TimeGAN with different loss functions
**This is a PyTorch implementation of the TimeGAN in "[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)" (Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, NIPS, 2019). We also add different loss functions to try the best performance of the frame work.**  

**My master-thesis is at the master branch.**   

## Introduction
這是一篇 TimeGAN 的不同 loss functions 的搭配實驗，目的在於找出最合適的 loss function 來減輕 mode collapse 的影響。

## TimeGAN architecture
請前往我的 [TimeGAN Pytorch 版本](https://github.com/kent1201/TimeGAN-Pytorch)觀看。

## Project architecture
`\Loss` 內含不同 train stages (encoder, decoder, generator, ...) 會用到的 loss。  

`\Network` 包含 `encoder.py`, `decoder.p`y, `generator.py`, `supervisor.py`, `discriminator.py` 五個不同部件，每個部件可用 rnn. lstm, gru, tcn 所替換。simple_discriminator.py, simple_predictor.py 則是用來評估 real data 與 syntheitc data 之間的差異所用的架構。 

`Configure.ini` 所有參數的設定檔。 

`requirements.txt` 套件要求。

`utils.py` 包含 train test data loader, random generator 等功能。  

`dataset.py, dataset_preprocess.py` 針對 Action base Datasets 的 Pytorch 操作。目前不提供 datasets。  

`train.py` TimeGAN 訓練部分。訓練部分分成三個階段: 
* `Stage 1` 訓練 encoder, decoder。
* `Stage 2` 訓練 encoder, supervisor, generator。
* `Stage 3` 聯合訓練 discriminator, generator, supervisor, encoder, decoder。  
訓練好模型將以日期作為劃分，儲存模型。

`generate_data.py` 經由訓練好的模型產生 synthetic data。 

`data_visualization.py` 對 real data 與 synthetic data 做 PCA, t-SNE 二維可視化圖。 

`test.py` 使用 discriminator 評估 real data 與 synthetic data 相似度 (以錯誤率當標準，越低越好)。使用 predictor 對 synthetic data 進行訓練，並在 real data 上進行預測(以 MAE 做標準，越低越好)。詳細標準可參考 "[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)"。 

`hyper_optimize.py` 開發測試中的功能，用於參數最佳化。

