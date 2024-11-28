import numpy as np
from tf_keras.models import load_model
from sklearn.metrics import confusion_matrix,accuracy_score

X_test = np.load('X_test1.npy')
y_test = np.load('y_test1.npy')

model = load_model('training_data1.h5')
yhat = model.predict(X_test)
print(yhat)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(ytrue)
print(yhat)
print(confusion_matrix(ytrue,yhat))
print(accuracy_score(ytrue,yhat))