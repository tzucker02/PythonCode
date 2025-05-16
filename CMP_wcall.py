# CMP_wcall.py
#
#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import math
import sys

import read_call_copy

# Family is = 1
# Alone is = 0

# df_new_copy['Alone_or_family'] = df_new_copy.parch + df_new.sibsp
# df_new_copy.loc[df_new_copy['Alone_or_family'] > 0, 'Alone_or_family'] = 1
# df_new_copy.loc[df_new_copy['Alone_or_family'] == 0, 'Alone_or_family'] = 0

numeric_columns = df_new_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
titanic_numeric = df_new_copy[numeric_columns]

correlation_matrix = titanic_numeric.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using matplotlib
plt.figure(figsize=(10, 8))
cmap = plt.cm.RdBu_r

# Plot the heatmap
im = plt.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im)

plt.title('Titanic Dataset Correlation Matrix', fontsize=14)
plt.xticks(np.arange(len(numeric_columns)), numeric_columns, rotation=45, ha='right')
plt.yticks(np.arange(len(numeric_columns)), numeric_columns)

# set and print the plot
plt.tight_layout()
plt.show()
plt.close()