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
df_new.describe()

feature = input("Which feature do you want to compare against the target? ")
print("The target is survival, and the feature to compare it to is " + feature)
# entry test, if nothing has been entered, set the feature to age
if feature == "":
    print("You did not enter a feature, so the feature will be set to the age of the passenger.")
    feature = 'age'
elif feature == None:
    print("You did not enter a feature, so the feature will be set to the age of the passenger.")
    feature = 'age'
elif feature not in df_new.columns:
    print("You entered " + feature + " which is not a column in the dataset. The feature will default to age.")
    # print("You did not enter a feature in the dataset, so the feature will be set to the age of the passenger.")
    feature = 'age'

# # boxplot distribution by survival
survived_class = df_new[df_new['target'] == 1][feature].dropna()
died_class = df_new[df_new['target'] == 0][feature].dropna()

plt.figure(figsize=(8,10))
plt.boxplot([died_class, survived_class], tick_labels=['Died (0)', 'Survived (1)'])
plt.title(feature.upper() + ' Distribution by Survival Status')
plt.xlabel('Survival Status')
plt.ylabel(feature.upper())
plt.grid(True, alpha=.25)
plt.show()
plt.close()