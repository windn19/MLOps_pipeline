import joblib

from sklearn.linear_model import LinearRegression

from model_preprocessing import preprocessing_data


(xtrain, ytrain), (xtest, ytest) = preprocessing_data('train.npy', 'test.npy')
lr = LinearRegression()
lr.fit(xtrain, ytrain)

joblib.dump(lr, 'model.jlib')
print('Model save')
