# plotsurvival.py
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

plt.figure(figsize=(8,10))
plt.boxplot([died_ages, survived_ages], tick_labels=['Died (0)', 'Survived (1)'])
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survival Status')
plt.ylabel('Age')
plt.grid(True, alpha=.25)

# save the plot as an image file
plt.savefig('Boxplot_Age_by_Survival.png')
# show the plot
plt.show()
# close the plot to regain memory resources
plt.close()