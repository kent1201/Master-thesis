# Multi-domain generative adversarial network learning for time-series data generation

Time-series data 一般是指具有時間維度的數據，如天氣記錄、支出記錄甚至股票趨勢。這些不同類型的數據遍布我們的生活，深刻影響著我們的行為。Time-series data 的預測也是深度學習領域的一項重要任務。但是 time-series data 需要隨著時間的推移進行記錄，這意味著很難在短時間內收集到足夠的資料。數據的缺乏也是深度學習領域的問題之一。因此，我們設計了一個基於 {\it Time-Series Generative Adversarial Network}~\cite{R1} 的框架，該方法約束了多種領域以保證模型的穩定性，減少模式崩潰的影響並提高生成數據的質量。


## Introduction
這是一篇 TimeGAN 的不同 loss functions 的搭配實驗，目的在於找出最合適的 loss function 來減輕 mode collapse 的影響。

## TimeGAN architecture
請前往我的 [TimeGAN Pytorch 版本](https://github.com/kent1201/TimeGAN-Pytorch)觀看。

## Project architecture
`\Loss` 內含不同 train stages (encoder, decoder, generator, ...) 會用到的不同 loss。loss 方面包含 supervised loss 與 unsupervised loss 結合以進行訓練。其中 supervised loss 分為 MSE loss 與 Soft-DTW loss。Unsupervised loss 則分為 Binary cross entropy with logits，WGAN-GP，Hinge loss。共可搭配出六種不同的方案。

`\Network` 包含 `encoder.py`, `decoder.p`y, `generator.py`, `supervisor.py`, `discriminator.py` 五個不同部件，每個部件可用 rnn. lstm, gru, tcn 所替換。simple_discriminator.py, simple_predictor.py 則是用來評估 real data 與 syntheitc data 之間的差異所用的架構。 

`Configure.ini` 所有參數的設定檔。參數可設定所需要的部件，supervised loss 與 unsupervised loss 的設定，dataset 路徑，batch size，training epoch，learning rate 等等。其餘包含產生資料，二維可視化的設定與測試的設定。 

`requirements.txt` conda 環境套件要求。

`utils.py` 包含 train test data loader, random generator 等功能。  

`dataset.py, dataset_preprocess.py` 針對 Action base Datasets 的 Pytorch 操作。不提供 datasets。 

`generate_data.py` 經由訓練好的模型產生 synthetic data。 

`data_visualization.py` 對 real data 與 synthetic data 做 PCA, t-SNE 二維可視化圖。 

`test.py` 使用 discriminator 評估 real data 與 synthetic data 相似度 (以錯誤率當標準，越低越好)。使用 predictor 對 synthetic data 進行訓練，並在 real data 上進行預測(以 MAE 做標準，越低越好)。詳細標準可參考 "[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)"。

**以下皆針對一般 time-series dataset 的 code 進行說明**

**以 "Time" 為首的 code 皆為針對一般 time-series dataset 使用。**

`Timedataset.py` 針對一般 time-series dataset 使用。

`Time_data_visualization.py` 二維可視化 real data 與 synthetic data 的分布。

`Time_generate_data.py` 使用訓練好的 model 來產生資料。

`Timetest.py` 對 real data 與 synthetic data 進行測試，測試內容包括 Classification score 與 Predictive score。

`train.py` TimeGAN 訓練部分。訓練部分分成三個階段: 
* `Stage 1` 訓練 encoder, decoder。
* `Stage 2` 訓練 encoder, supervisor, generator。
* `Stage 3` 聯合訓練 discriminator, generator, supervisor, encoder, decoder。  
訓練好模型將以日期作為劃分，儲存模型。 

`hyper_optimize.py` 開發測試中的功能，用於參數最佳化。

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

