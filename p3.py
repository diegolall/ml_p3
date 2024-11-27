from toy_script import load_data

if __name__ == '__main__':

    #Data importation
    X_train, y_train, X_test = load_data("./")
    print(X_train[3500][0])