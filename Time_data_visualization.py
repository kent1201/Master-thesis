"""
visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import configparser
import os
from dataset_preprocess import MinMaxScaler1, batch_generation

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

synthetic_dataset_dir = config.get('GenTstVis', 'syntheticDataset_path') + '/' + config.get('GenTstVis', 'date_dir') + \
    '/' + config.get('GenTstVis', 'classification_dir') + '/' + \
    config.get('GenTstVis', 'synthetic_data_name')

real_dataset_dir = config.get('GenTstVis', 'Dataset_path')

pic_path = config.get('GenTstVis', 'pic_path') + '/' + config.get('GenTstVis',
                                                                  'date_dir') + '/' + config.get('GenTstVis', 'classification_dir')

seq_len = config.getint('GenTstVis', 'seq_len')

pca_pic_name = config.get('data_visualization', 'pca_pic_name')

t_sne_pic_name = config.get('data_visualization', 't_sne_pic_name')


def visualization(ori_data, generated_data, analysis, pic_path, pic_name):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    # idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    idx = np.round(np.linspace(0, len(ori_data)-1, anal_sample_no)).astype(int)

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    # ori_data = ori_data[:anal_sample_no]
    # generated_data = generated_data[:anal_sample_no]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            # 對 each row 求均值 => [82, 1]. reshape to [1, 82]
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + \
        ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        # fit（prep_data），表示用數據 prep_data 來訓練PCA模型。
        pca.fit(prep_data)
        # 數據 prep_data 轉換成降維後的數據。當模型訓練好後，對於新輸入的數據，都可以用 transform 方法來降維。
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        # plt.subplots(1) 只生成 1x1 的子圖 (只有一張圖)
        f, ax = plt.subplots(1)
        # alpha: 透明度
        # plt.scatter: 散佈圖
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        # ax.legend(): 顯示圖例
        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.savefig(pic_path + "/" + pic_name, bbox_inches='tight')
        # plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        # random_state=0, init='pca', method='barnes_hut', angle=0.5)
        tsne = TSNE(n_components=2, verbose=1, perplexity=45, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(pic_path + "/" + pic_name, bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':

    real_data = np.loadtxt(real_dataset_dir, delimiter=",", skiprows=0)
    # real_data = real_data[::-1]
    real_data, _, _ = MinMaxScaler1(real_data)
    batch_real_data = batch_generation(real_data, seq_len, 1)
    batch_real_data = batch_real_data[:-1]

    synthetic_data = np.loadtxt(
        synthetic_dataset_dir, delimiter=",", skiprows=0)
    # synthetic_data = synthetic_data[::-1]
    synthetic_data, _, _ = MinMaxScaler1(synthetic_data)
    batch_synthetic_data = []
    batch_synthetic_data = batch_generation(synthetic_data, seq_len, seq_len)

    min_batch_len = len(batch_synthetic_data) if len(batch_synthetic_data) < len(batch_real_data) else len(batch_real_data)

    batch_synthetic_data = batch_synthetic_data[:min_batch_len]
    batch_real_data = batch_real_data[:min_batch_len]

    print("batch_synthetic_data: {}".format(len(batch_synthetic_data)))
    print("batch_real_data: {}".format(len(batch_real_data)))
    

    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    visualization(batch_real_data, batch_synthetic_data,
                  'pca', pic_path, pca_pic_name)
    visualization(batch_real_data, batch_synthetic_data,
                  'tsne', pic_path, t_sne_pic_name)
