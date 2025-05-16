import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import math
import sys

file_path = input("Full path to dataframe: ")
file_separator = input("Separator used in dataframe (i.e., tab): ")
target_col = input("Target feature in the dataframe (column name): ")

def read_file(filepath=None, file_separator=None, target=None):

    # check for valid inputs and set defaults
    if filepath is None:
        print("Please provide a file path.")
        sys.exit(1)
    elif filepath.upper() == "TITANIC":
        filepath = "https://github.com/EpistasisLab/pmlb/raw/refs/heads/master/datasets/titanic/titanic.tsv.gz"
    
    if file_separator is None:  
        print("Please provide a file separator.")
        sys.exit(1)
    if file_separator.upper() == "T": 
        file_separator = "\t"
    elif file_separator.upper() == "C":
        file_separator = ","
    elif file_separator.upper() == "SPACE":
        file_separator = " "
    elif file_separator.upper() == "SEMICOLON":
        file_separator = ";"
    elif file_separator.upper() == "TAB":
        file_separator = "\t"
    elif file_separator.upper() == "PIPE":
        file_separator = "|"
    elif file_separator.upper() == "COLON":
        file_separator = ":"
    elif file_separator.upper() == "NONE":
        file_separator = None
    elif file_separator.upper() == "OTHER":
        file_separator = input("Please provide a custom file separator: ")
     

    if target is None:
        print("Please provide a target variable.")
        sys.exit(1)

    # read the file
    df = pd.read_csv(filepath, sep=file_separator, engine='python')
    df_new = df.dropna()

    # dataframe = df_new
    # split dataset into features and target
    df_features = df_new.drop(columns=[target])
    df_target = df_new[target]
    
    return df_new, df_features, df_target


df, features, target_values = read_file(
    file_path,
    file_separator,
    target_col
)

# drop rows with missing data while preserving the original dataset
df_new = df.dropna()
df_new_copy = df_new.copy()

# Family is = 1
# Alone is = 0

df_new_copy['Alone_or_family'] = df_new_copy.parch + df_new.sibsp
df_new_copy.loc[df_new_copy['Alone_or_family'] > 0, 'Alone_or_family'] = 1
df_new_copy.loc[df_new_copy['Alone_or_family'] == 0, 'Alone_or_family'] = 0

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