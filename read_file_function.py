# read_file_function.py
#
#!/usr/bin/env python
# coding: utf-8

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
