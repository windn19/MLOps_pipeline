import joblib

from sklearn.metrics import r2_score, mean_absolute_error as mae

from model_preparation import preprocessing_data


(_, _), (xtest, ytest) = preprocessing_data('train.npy', 'test.npy')
model = joblib.load('model.jlib')
yscaler = joblib.load('yscaler.jlib')
pred = model.predict(xtest)
pred = yscaler.inverse_transform(pred)
ytrue = yscaler.inverse_transform(ytest)
print(f'Коэффициент детерминации: {r2_score(ytest, pred):.4f}')
print(f'Средняя абсолютная ошибка: {mae(ytest, pred):.4f}')
