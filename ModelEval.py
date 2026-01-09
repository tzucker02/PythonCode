import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import math
import sys
from IPython.display import display, Javascript
import zipfile

def read_file(filepath, file_separator):
    df = pd.read_csv(filepath, sep=file_separator, engine='python')
    df_new = df.dropna()
    return df_new

def choose_target_column(df):
    from pandas.api.types import is_numeric_dtype

    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise ValueError("No numeric columns found; please provide a dataset with at least one numeric target.")

    print("Available numeric columns (choose a target):")
    for idx, col in enumerate(numeric_cols, start=1):
        print(f"[{idx}] {col}")

    selection = input("Enter the number of the target column (default=1): ").strip()
    if selection == "":
        selection = 1
    try:
        selection = int(selection)
        target = numeric_cols[selection - 1]
    except (ValueError, IndexError):
        print("Invalid selection; defaulting to the first numeric column.")
        target = numeric_cols[0]

    return target

# Helper function to check if a file is a ZIP file

def is_zip_file(filename):
    return zipfile.is_zipfile(filename)

file_path = input("Full path to dataframe (zip files are not supported): ")
file_separator = input("Separator used in dataframe (i.e., '\\t' for tab, '\,' for comma, etc., without the quotes: ")

# extract zip file if needed
if is_zip_file(file_path):
    # importing required modules
    from zipfile import ZipFile

    # specifying the zip file name
    file_name = file_path

    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
    file_path = zip.namelist()[0]  # assuming we want the first file in the zip
    
try:
    dataframe = read_file(file_path,file_separator)
    target_col = choose_target_column(dataframe)
    # Split out features/target for convenience
    features = dataframe.drop(columns=[target_col])
    target_values = dataframe[target_col]
except:
    print("Error reading file. Please check the file path and separator. Please rerun this notebook and provide a link to a supported filetype.")
    print("Supported filetypes include CSV, TSV, and other text-based formats compatible with pandas read_csv.")
    exit()


def plot_to_compare(model_name1, score1, mse1, model_name2, score2, mse2, model_name3, score3, mse3, model_name4, score4, mse4):
    
    print(f"\n\n\033[1mThis can be plotted with the following results:\033[0m\n")
    
    models = [model_name1, model_name2, model_name3, model_name4]
    scores = [score1, score2, score3, score4]
    mse_scores = [mse1, mse2, mse3, mse4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    bars1 = ax1.bar(models, mse_scores, color=['skyblue', 'orange', 'green', 'red'])
    for bar, score in zip(bars1, mse_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    ax1.set_title('Model Mean Squared Error (LOWER IS BETTER)')
    ax1.set_ylabel('MSE')
    ax1.set_ylim(0, max(mse_scores) * 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    bars2 = ax2.bar(models, scores, color=['skyblue', 'orange', 'green', 'red'])
    for bar, score in zip(bars2, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    ax2.set_title('Model Accuracy (HIGHER IS BETTER)')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, max(scores) * 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15)

    plt.tight_layout()
    plt.show()
    plt.close()

    return

def train_and_evaluate_model(model_type, model_name_str, X_train, X_test, y_train, y_test):
    
    from sklearn.metrics import mean_squared_error

    model = model_type
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Model:\033[1m{model_name_str:<24}\033[0m\tAccuracy Score:\t\033[1m{score:.4f}\033[0m\tMSE Score:\t\033[1m{mse:.4f}\033[0m")
    
    return model_name_str, score, mse

def train_it(data, target):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import LinearSVC

    y = data[target].copy()
    X = data.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Call the train and evaluate model function for each model
    
    iters = 1000
    penalty = 'l2'
    estimators = 7
    depth = 7
    
    model_name1, score1, mse1 = train_and_evaluate_model(LogisticRegression(penalty=penalty, max_iter=iters), 'Logistic Regression', X_train, X_test, y_train, y_test)
    model_name2, score2, mse2 = train_and_evaluate_model(LinearSVC(penalty=penalty, max_iter=iters), 'Linear SVC', X_train, X_test, y_train, y_test)
    model_name3, score3, mse3 = train_and_evaluate_model(Ridge(alpha=1.0, max_iter=iters), 'Ridge', X_train, X_test, y_train, y_test)
    model_name4, score4, mse4 = train_and_evaluate_model(RandomForestClassifier(n_estimators=estimators, max_depth=depth), 'Random Forest Classifier', X_train, X_test, y_train, y_test)
    # Call the plot function with the results from the train and evaluate function

    plot_to_compare(model_name1, score1, mse1, model_name2, score2, mse2, model_name3, score3, mse3, model_name4, score4, mse4)

    return

train_it(dataframe, target_col)
