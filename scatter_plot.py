import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import math
import sys

df = pd.read_csv('https://github.com/EpistasisLab/pmlb/raw/refs/heads/master/datasets/titanic/titanic.tsv.gz', sep = "\t")
# drop rows with missing data while preserving the original dataset
df_new = df.dropna()

_ = pd.plotting.scatter_matrix(df_new, alpha=0.2, figsize=(20,20));