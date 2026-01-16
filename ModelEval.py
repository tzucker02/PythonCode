## This set of functions require the following inputs
"""
1. a complete path to a pandas dataframe (file_path)
2. the separator used in the data file - preceded by a backslash (file_separator):
    a. comma - \\,
    b. semi-colon - \\;
    c. colon - \\:
    d. etc.
3. the target to predict (target_col) - the target is assumed to be numeric

For Plotting the script asks for 
    a. plot titles for the axes
    b. plot labels
    c. Four(4) comma separated hex values for colors of the bars to be plotted
 
For all the above, you can press enter to retain the default values (shown)

The main code of this notebook looks like this:

*train_it(file_path, target_col)*

but other parts of the function require the file separator (file_separator) as well, in order to properly read in the file.

NOTE: I have attempted to make the code as easy to read as possible by naming the functions with descriptive titles.
"""

from logging import root
from unicodedata import name
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import math
import sys
from IPython.display import display, Javascript
import zipfile
from ast import If
import tkinter as tk
from tkinter import simpledialog, messagebox

def greet_user():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    name = simpledialog.askstring("Input", "Enter your name:")
    if name:
        messagebox.showinfo("Greeting", f"Hello, {name}!")  
    return   


# function to read in file based on type

def read_file(filepath, file_separator):
    
    if file_separator == None:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        file_type = simpledialog.askstring("Input", "Is your file fixed width(FW) or separated by a specific character(SC)? (FW/SC): ")
    else:
        file_type = 'SC'
    
    if file_type.upper() == 'FW':
        df = pd.read_fwf(filepath)
    else:
        df = pd.read_csv(filepath, sep=file_separator, engine='python')
        
    df_new = df.dropna()
    return df_new

def choose_target_column(df):
    from pandas.api.types import is_numeric_dtype

    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        messagebox.showinfo("Error", "No numeric columns found; please provide a dataset with at least one numeric target.")
        # raise ValueError("No numeric columns found; please provide a dataset with at least one numeric target.")
    root = tk.Tk()
    root.withdraw()  # Hide main window
    # print("Available numeric columns (choose a target - if your target is NOT numeric, your dataset may not be suitable):")
    dict_numeric_cols = {}
    # for idx, col in enumerate(numeric_cols, start=1):
    #    dict_numeric_cols[str(idx)] = col
        # messagebox.showinfo("Numeric Columns", f"[{idx}] {col}")
        # print(f"[{idx}] {col}")
    messagebox.showinfo("Numeric Columns", "\n".join([f"[{idx}] {col}" for idx, col in enumerate(numeric_cols, start=1)]))
    
    selection = simpledialog.askstring("Input", "Enter the number of the target column (default=1 - if your target is NOT numeric, your dataset may not be suitable): ").strip()
    if selection == "":
        selection = 1
    try:
        selection = int(selection)
        target = numeric_cols[selection - 1]
    except (ValueError, IndexError):
        messagebox.showinfo("Error", "Invalid selection; defaulting to the first numeric column.")
        # print("Invalid selection; defaulting to the first numeric column.")
        target = numeric_cols[0]

    return target

# Helper function to check if a file is a ZIP file

def is_zip_file(filename):
    return zipfile.is_zipfile(filename)

root = tk.Tk()
root.withdraw()  # Hide main window

file_path = simpledialog.askstring("Input", "Full path to dataframe (zip files are not supported): ")
file_separator = simpledialog.askstring("Input", "Separator used in dataframe (i.e., '\\t' for tab, '\,' for comma, etc., without the quotes: ")

if is_zip_file(file_path):
    messagebox.showinfo("Error", "ZIP files are not supported. Please provide a direct path to a supported filetype.")
    # print("ZIP files are not supported. Please provide a direct path to a supported filetype.")
    exit()
    
try:
    dataframe = read_file(file_path,file_separator)
    target_col = choose_target_column(dataframe)
    # Split out features/target for convenience
    features = dataframe.drop(columns=[target_col])
    target_values = dataframe[target_col]
except:
    messagebox.showinfo("Error", "Error reading file. Please check the file path and separator. Please rerun this notebook and provide a link to a supported filetype.")
    exit()


def plot_to_compare(model_name1, score1, mse1, model_name2, score2, mse2, model_name3, score3, mse3, model_name4, score4, mse4):
    
    # Define axis titles and labels
    root = tk.Tk()
    root.withdraw()  # Hide main window
    Axis1Title = simpledialog.askstring("Input", "Enter title for MSE plot: (Default: 'Model Mean Squared Error (LOWER IS BETTER)'): ")
    if Axis1Title.strip() == "":
        Axis1Title = "Model Mean Squared Error (LOWER IS BETTER)"
    Axis2Title = simpledialog.askstring("Input", "Enter title for Accuracy plot: (Default: 'Model Accuracy (HIGHER IS BETTER)'): ")
    if Axis2Title.strip() == "":
        Axis2Title = "Model Accuracy (HIGHER IS BETTER)"
    Axis1_YLabel = simpledialog.askstring("Input", "Enter label for MSE Y-axis (Default: 'Mean Squared Error'): ")
    if Axis1_YLabel.strip() == "":
        Axis1_YLabel = "Mean Squared Error"
    Axis2_YLabel = simpledialog.askstring("Input", "Enter label for Accuracy Y-axis (Default: 'Accuracy'): ")
    if Axis2_YLabel.strip() == "":
        Axis2_YLabel = "Accuracy"
        
    # Define color palette
    colorpalette = simpledialog.askstring("Input", "Enter 4 hex color codes separated by commas for the bars (Default: '#4E79A7 - skyblue,#F28E2B - orange,#E15759 - red,#76B7B2 - teal'): ")   
    if colorpalette.strip() == "":
        colorpalette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
    else:
        colorpalette = [color.strip() for color in colorpalette.split(",")]
    
    print(f"\n\n\033[1mThis can be plotted with the following results:\033[0m\n")
    
    models = [model_name1, model_name2, model_name3, model_name4]
    scores = [score1, score2, score3, score4]
    mse_scores = [mse1, mse2, mse3, mse4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    bars1 = ax1.bar(models, mse_scores, color=colorpalette)
    for bar, score in zip(bars1, mse_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    ax1.set_title(Axis1Title)
    ax1.set_ylabel(Axis1_YLabel)
    ax1.set_ylim(0, max(mse_scores) * 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    bars2 = ax2.bar(models, scores, color=colorpalette)
    for bar, score in zip(bars2, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    ax2.set_title(Axis2Title)
    ax2.set_ylabel(Axis2_YLabel)
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