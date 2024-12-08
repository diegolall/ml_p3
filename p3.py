from toy_script import load_data
from pre_process import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X_train_unprocessed, y_train_unprocessed, X_test_unprocessed = load_data(data_path='./')
    
    #preprocess (skidable if the data is already preprocessed)
    """
    titles_table, X_train_preprocess = preprocess_data(raw_data=X_train, threshold_missing_points=0.05,
                                                        titles=True, frequencies_number=5,threshold_freq=1e-6)
    print(X_train_preprocess)
    
    fichier = "preprocess_test_16freq.txt"

    # Écriture dans le fichier
    with open(fichier, "w") as file:
        # Parcourir les lignes du tableau
        file.write("\t".join(titles_table) + "\n")
        for row in X_train_preprocess:
            line = "\t".join(map(str, row))
            file.write(line + "\n")"""
    #if data is already preprocessed
    file_path = "preprocess_data_train_15freq.txt"
    #import X data preprocessed
    X = pd.read_csv(file_path, sep="\t")
    #droping frequencies (keeping only one frequency)
    n_freq = 1
    X_train = drop_frequencies(X,n_freq)
    y_train = y_train_unprocessed
    
    print("Data loaded and preprocessed with frequencies dropped \nX_train shape: ", X_train.shape, " y_train shape: ", y_train.shape, "\nNumber of frequencies: ", n_freq,"\nReady to train the model")
    
    