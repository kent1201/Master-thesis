# Multi-domain generative adversarial network learning for time-series data generation

這邊是我的論文的程式碼。詳細內容請翻閱我的論文 [Multi-domain generative adversarial network learning for time-series data generation](https://github.com/kent1201/Master-thesis/blob/master/Src/Multi-domain%20generative%20adversarial%20network%20learning%20for%20time-series%20data%20generation%20v3.pdf).

## Introduction
Time-series data 一般是指具有時間維度的數據，如天氣記錄、支出記錄甚至股票趨勢。這些不同類型的數據遍布我們的生活，深刻影響著我們的行為。Time-series data 的預測也是深度學習領域的一項重要任務。但是 time-series data 需要隨著時間的推移進行記錄，這意味著很難在短時間內收集到足夠的資料。數據的缺乏也是深度學習領域的問題之一。因此，我們設計了一個基於 [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)" (Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, NIPS, 2019) 的 framework，該方法約束了多種領域以保證模型的穩定性，減少模式崩潰的影響並提高生成數據的質量。本篇方法以 TimeGAN 的概念為基礎，除了專注在高維空間的訓練，更近一步的針對 data 的 domain 與 noise 的 domain 進行訓練，強化了 data 在不同 domain 之間的聯繫，使模型能更完整的學習 data 的特徵與時間動態。

## System framework

以下是我們方法的架構圖。

![System framework](https://github.com/kent1201/Master-thesis/blob/master/Src/System%20Framework%20new.png)


## Project architecture
`\Loss` 內含不同 train stages (encoder, decoder, generator, ...) 會用到的不同 loss。loss 方面包含 supervised loss 與 unsupervised loss 結合以進行訓練。其中 supervised loss 分為 MSE loss 與 Soft-DTW loss。Unsupervised loss 則分為 Binary cross entropy with logits，WGAN-GP，Hinge loss。共可搭配出六種不同的方案。我的論文方法目前以 MSE + Hinge loss 組合來進行實驗。

`Loss_curve` 包含訓練與測試(訓練分類與預測模型)時的 loss 曲線。

`\Network` 包含 `embedder.py`, `recovery.py`, `generator.py`, `supervisor.py`, `discriminator.py` 五個不同部件，每個部件可用 rnn. lstm, gru, tcn 所替換。simple_discriminator.py, simple_predictor.py 則是用測試時用來評估 real data 與 syntheitc data 之間的差異所用的架構。[c_rnn_gan.py](https://github.com/olofmogren/c-rnn-gan) 則是用來比較的方法。

`data` 內含目前用到的 real datasets。

`Configure.ini` 所有參數的設定檔。參數可設定所需要的部件，supervised loss 與 unsupervised loss 的設定，dataset 路徑，batch size，training epoch，learning rate 等等。其餘包含產生資料，二維可視化的設定與測試的設定。 

`Time_data_visualization.py` 二維可視化 real data 與 synthetic data 的分布。

`Time_generate_data.py` 使用訓練好的 model 來產生資料。

`Timedataset.py` 針對一般 time-series dataset 使用。

`Timetest.py` 訓練一個簡單的(兩層 GRU) discriminator 來評估 real data 與 synthetic data 相似度 (以偏差率當標準，越低越好)。訓練一個簡單的(五層 GRU) predictor 對 synthetic data 進行訓練，並在 real data 上進行預測(以 MAE 做標準，越低越好)。

`c_rnn_gan_generate_data.py` 針對 c_rnn_gan 模型來生成資料。

`dataset_preprocess.py` 載入 dataset 時對其進行前處理。

`requirements.txt` conda 環境套件要求。

`train.py` 訓練模型部分。訓練分成三個階段: 
* `Stage 1` 預訓練 data autoencoder。
* `Stage 2` 預訓練 supervisor。
* `Stage 3` 聯合訓練。  
訓練好模型將以日期作為劃分，儲存模型。 

`train_c_rnn_gan.py` 訓練 c_rnn_gan 模型做為比較對象。

`utils.py` 包含 train test data loader, random generator 等功能。 

## Requirements

* conda 4.8.2
```bash
conda install --yes --file requirements.txt
``` 
or
```bash
pip install -r requirements.txt
```

### How to use

>Set the Configure.ini
>conda create your environment 
>
>conda activate your environment 
>
>pip install the requirments 
```python
python train.py
python Time_generate_data.py
python Time_data_visualization.py
python Timetest.py
```
* **Notice** 無提供 Dataset，請自行根據使用的 dataset 自行調整程式內容。

