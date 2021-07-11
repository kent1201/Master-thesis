import math
import torch
import torch.nn as nn
import configparser
import os
import pandas as pd
from datetime import date
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from dataset_preprocess import MinMaxScaler1, batch_generation
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default',
                                               'cuda_device_number') if torch.cuda.is_available() else "cpu")

synthetic_dataset_dir = config.get('GenTstVis', 'syntheticDataset_path') + '/' + config.get('GenTstVis', 'date_dir') + \
    '/' + config.get('GenTstVis', 'classification_dir') + '/' + \
    config.get('GenTstVis', 'synthetic_data_name')

real_dataset_dir = config.get('GenTstVis', 'Dataset_path')


d_num_epochs = config.getint('test', 'd_num_epochs')
p_num_epochs = config.getint('test', 'p_num_epochs')
batch_size = config.getint('test', 'batch_size')
learning_rate = config.getfloat('test', 'learning_rate')
seq_len = config.getint('GenTstVis', 'seq_len')
PADDING_VALUE = config.getfloat('default', 'padding_value')
test_iteration = config.getint('test', 'test_iteration')

dis_curve = 0
pred_curve = 0


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

def batch_generator(data, time, batch_size):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]     
            
  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)
  
  return X_mb, T_mb


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def discriminative_score_metrics (ori_data, generated_data):
  """Use post-hoc RNN to classify original data and synthetic data
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
  """
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape    
    
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
  ## Builde a post-hoc RNN discriminator network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 2000
  batch_size = 128
    
  # Input place holders
  # Feature
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  X_hat = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
    
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")
  T_hat = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t_hat")

  # discriminator function
  def discriminator (x, t):
    """Simple discriminator function.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat_logit: logits of the discriminator output
      - y_hat: discriminator output
      - d_vars: discriminator variables
    """
    with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE) as vs:
      d_cell = tf.keras.layers.GRUCell(hidden_dim, activation='tanh', name = 'd_cell')
      d_rnn = tf.keras.layers.RNN(d_cell,return_sequences=True,return_state=True)
      d_outputs,d_last_states = d_rnn(x)
      y_hat_logit = tf.keras.layers.Dense(1, activation=None)(d_last_states)
      y_hat = tf.nn.sigmoid(y_hat_logit)
      d_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]
    
    return y_hat_logit, y_hat, d_vars
    
  y_logit_real, y_pred_real, d_vars = discriminator(X, T)
  y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
  # Loss for the discriminator
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
                                                                       labels = tf.ones_like(y_logit_real)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
                                                                       labels = tf.zeros_like(y_logit_fake)))
  d_loss = d_loss_real + d_loss_fake
    
  # optimizer
  d_solver = tf.compat.v1.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
  ## Train the discriminator   
  # Start session and initialize
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # Train/test division for both original and generated data
  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
  train_test_divide(ori_data, generated_data, ori_time, generated_time)

  dis_training_loss_list = []
    
  # Training step
  for itt in range(iterations):
          
    # Batch setting
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    # print("discriminative- {}:{}".format(itt, np.asarray(X_mb).shape))
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
          
    # Train discriminator
    _, step_d_loss = sess.run([d_solver, d_loss], 
                              feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})   
    
    dis_training_loss_list.append(step_d_loss) 

  plt.plot(dis_training_loss_list, color='red')
  plt.title("Discriminative training loss")
  plt.xlabel('iteration')
  plt.savefig('/home/kent1201/Documents/Master-thesis/Loss_curve/dis_loss_curve.png', bbox_inches='tight')
  plt.close()        
    
  ## Test the performance on the testing set    
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
  # Compute the accuracy
  acc = accuracy_score(y_label_final, (y_pred_final>0.5))
  discriminative_score = np.abs(0.5-acc)
    
  return discriminative_score 

def predictive_score_metrics (ori_data, generated_data):
  """Report the performance of Post-hoc RNN one-step ahead prediction.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - predictive_score: MAE of the predictions on the original data
  """
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
  ## Builde a post-hoc RNN predictive network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128
    
  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-2], name = "myinput_x")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")    
  Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")
    
  # Predictor function
  def predictor (x, t):
    """Simple predictor function.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat: prediction
      - p_vars: predictor variables
    """
    with tf.compat.v1.variable_scope("predictor", reuse = tf.compat.v1.AUTO_REUSE) as vs:
      p_cell = tf.keras.layers.GRUCell(hidden_dim, activation='tanh', name = 'p_cell')
      p_outputs= tf.keras.layers.RNN(p_cell,return_sequences=True)(x)
      y_hat_logit = tf.keras.layers.Dense(1,activation=None)(p_outputs)
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]
    
    return y_hat, p_vars
    
  y_pred, p_vars = predictor(X, T)
  # Loss for the predictor
  p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
  # optimizer
  p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
        
  ## Training    
  # Session start
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # Training using Synthetic dataset
  for itt in range(iterations):
          
    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]     
            
    X_mb = list(generated_data[i][:-1,:(dim-2)] for i in train_idx)
    T_mb = list(generated_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:,(dim-2)],[len(generated_data[i][1:,(dim-2)]),1]) for i in train_idx)        
          
    # Train predictor
    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        
    
  ## Test the trained model on the original data
  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]
    
  X_mb = list(ori_data[i][:-1,:(dim-2)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:,(dim-2)], [len(ori_data[i][1:,(dim-2)]),1]) for i in train_idx)
    
  # Prediction
  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
    
  # Compute the performance in terms of MAE
  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
  predictive_score = MAE_temp / no
    
  return predictive_score


if __name__ == '__main__':

    real_data = np.loadtxt(real_dataset_dir, delimiter=",", skiprows=0)
    # real_data = real_data[::-1]
    real_data, _, _ = MinMaxScaler1(real_data)
    batch_real_data = batch_generation(real_data, seq_len, 1)
    # batch_real_data = batch_real_data[:-1]

    synthetic_data = np.loadtxt(synthetic_dataset_dir, delimiter=",", skiprows=0)
    # synthetic_data = synthetic_data[::-1]
    synthetic_data, _, _ = MinMaxScaler1(synthetic_data)
    batch_synthetic_data = []
    batch_synthetic_data = batch_generation(synthetic_data, seq_len, seq_len)

    min_batch_len = len(batch_synthetic_data) if len(batch_synthetic_data) < len(batch_real_data) else len(batch_real_data)

    batch_synthetic_data = batch_synthetic_data[:min_batch_len]
    batch_real_data = batch_real_data[:min_batch_len]
    
    discriminative_score_list = []
    predictive_score_list = []


    for iteration in range(0, test_iteration):
        
        discriminative_score = discriminative_score_metrics(batch_real_data, batch_synthetic_data)
        predictive_score = predictive_score_metrics(batch_real_data, batch_synthetic_data)
        
        # print("iteration: {}, predictive_score: {:.6f}".format(iteration, predictive_score))
        print("iteration: {}, discriminative_score: {:.6f}, predictive_score: {:.6f}".format(iteration, discriminative_score, predictive_score))

        discriminative_score_list.append(discriminative_score)
        predictive_score_list.append(predictive_score)
    
    mean_discriminative_score = np.mean(discriminative_score_list)
    mean_predictive_score = np.mean(predictive_score_list)

    print("Discriminative score: {:.4f}".format(mean_discriminative_score))
    print("Predictive score: {:.4f}".format(mean_predictive_score))
