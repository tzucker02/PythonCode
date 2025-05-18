# survivedpassengershistogram.py
#
#!/usr/bin/env python
# coding: utf-8

import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model

# read in the dataset
df = pd.read_csv('https://github.com/EpistasisLab/pmlb/raw/refs/heads/master/datasets/titanic/titanic.tsv.gz', sep = "\t")
# drop rows with missing data while preserving the original dataset
df_new = df.dropna()

survived_ages = df_new[df_new['target'] == 1]['age'].dropna()
died_ages = df_new[df_new['target'] == 0]['age'].dropna()

plt.figure(figsize=(10, 5))
plt.hist(survived_ages, bins=80, edgecolor='black')
plt.title('Distribution of Age of Survivors')
plt.xlabel('Age of Passengers')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('DistributionOfAgeOfSurvivors.png')
plt.show()
plt.close()