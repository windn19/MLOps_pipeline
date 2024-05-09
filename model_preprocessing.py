import joblib

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocessing_data(train_file, test_file):
    train, test = np.load(train_file), np.load(test_file)
    xtrain, ytrain = train[:, :9], train[:, 9]
    xtest, ytest = test[:, :9], test[:, 9]
    xscaler = StandardScaler()
    yscaler = MinMaxScaler()
    X = np.concatenate([xtrain, xtest], axis=0)
    y = np.concatenate([ytrain, ytest], axis=0)
    yscaler.fit(y.reshape(-1, 1))
    xscaler.fit(X)
    xtrain, ytrain = xscaler.transform(xtrain), yscaler.transform(ytrain.reshape(-1, 1))
    xtest, ytest = xscaler.transform(xtest), yscaler.transform(ytest.reshape(-1, 1))
    joblib.dump(yscaler, 'yscaler.jlib')
    return (xtrain, ytrain), (xtest, ytest)
