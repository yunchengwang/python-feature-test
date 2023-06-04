#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[2]:


## Relevant feature test

def plot_mos_density(features, dim, mos, bins=32, metric='mse'):
    density_map = np.zeros((100, bins))
    mos = np.floor(mos).astype(int)
    f_1d = features[:, dim]
    f_1d, mos = remove_outliers(f_1d, mos)
    f_min, f_max = f_1d.min(), f_1d.max()
    bin_width = (f_max - f_min) / bins
    for i in range(len(mos)):
        density_map[mos[i], np.floor((f_1d[i] - f_min - 0.0001)/bin_width).astype(int)] += 1
        
    best_error, best_partition_index, l_mean, r_mean = find_best_partition(f_1d, mos, bins, metric=metric)
    plt.figure()
    plt.axvline(x=best_partition_index, color='red', label='partition point')
    plt.axhline(y=l_mean, xmin=0, xmax=best_partition_index/bins, color='yellow', label='left mean')
    plt.axhline(y=r_mean, xmin=best_partition_index/bins, xmax=1, color='white', label='right mean')
    plt.legend()
    plt.imshow(density_map)


def find_best_partition(f_1d, mos, bins=32, metric='mse'):
    f_1d, mos = remove_outliers(f_1d, mos)
    best_error = float('inf')
    best_partition_index = 0
    left_mean, right_mean = 0, 0
    f_min, f_max = f_1d.min(), f_1d.max()
    bin_width = (f_max - f_min) / bins
    for i in range(1, bins):
        partition_point = f_min + i * bin_width
        left_mos, right_mos = mos[f_1d <= partition_point], mos[f_1d > partition_point]
        partition_error = get_partition_error(left_mos, right_mos, metric)
        if partition_error < best_error:
            best_error = partition_error
            best_partition_index = i
            left_mean = left_mos.mean()
            right_mean = right_mos.mean()
    return best_error, best_partition_index, left_mean, right_mean
        

def get_partition_error(left_mos, right_mos, metric='mse'):
    if metric == 'mse':
        n1, n2 = len(left_mos), len(right_mos)
        left_mse = ((left_mos - left_mos.mean())**2).sum()
        right_mse = ((right_mos - right_mos.mean())**2).sum()
        return np.sqrt((left_mse + right_mse) / (n1 + n2))
    else:
        print('Unsupported error')
        return 0


def remove_outliers(f_1d, mos, n_std=2.0):
    # f is a 1D feature
    new_f_1d = []
    new_mos = []
    f_mean, f_std = f_1d.mean(), f_1d.std()
    for i in range(len(mos)):
        if np.abs(f_1d[i] - f_mean) <= n_std * f_std:
            new_f_1d.append(f_1d[i])
            new_mos.append(mos[i])
    return np.array(new_f_1d), np.array(new_mos)

## High-dimensional RFT

def find_best_partition_ho(f_ho, mos, bins=32, metric='mse'):
    if f_ho.shape[1] == 1:
        f_1d = f_ho.reshape(-1)
    else:
        pca = PCA(n_components=1)
        f_1d = pca.fit_transform(f_ho).reshape(-1)
    f_1d, mos = remove_outliers(f_1d, mos)
    best_error = float('inf')
    best_partition_index = 0
    left_mean, right_mean = 0, 0
    f_min, f_max = f_1d.min(), f_1d.max()
    bin_width = (f_max - f_min) / bins
    for i in range(1, bins):
        partition_point = f_min + i * bin_width
        left_mos, right_mos = mos[f_1d <= partition_point], mos[f_1d > partition_point]
        partition_error = get_partition_error(left_mos, right_mos, metric)
        if partition_error < best_error:
            best_error = partition_error
            best_partition_index = i
            left_mean = left_mos.mean()
            right_mean = right_mos.mean()
    return best_error, best_partition_index, left_mean, right_mean


# In[3]:


## Synthetic datasets: 1000 samples, 100 features; labels is uniform distributed in [1, 5]
### train RFT
X = np.random.randn(1000, 100)
y = np.random.uniform(1, 5, 1000)

dim_mse = dict()
for d in range(X.shape[1]):
    best_mse, _, _, _ = find_best_partition(X[:, d], y, bins=32)
    dim_mse[d] = best_mse
    
sorted_mse = {k: v for k, v in sorted(dim_mse.items(), key=lambda item: item[1])}

plt.figure()
plt.plot(np.arange(100), list(sorted_mse.values()))
plt.xlabel('Rank')
plt.ylabel('RMSE')
plt.show()

### Extract relevant features
relevant_dimension = np.array(list(sorted_mse.keys()))[np.arange(20)]
selected_X = X[:, relevant_dimension]
print(selected_X.shape)


# In[ ]:




