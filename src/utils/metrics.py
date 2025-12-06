from sklearn.metrics import accuracy_score as _acc
from sklearn.metrics import f1_score

def accuracy_score(y_true, y_pred):
    return _acc(y_true, y_pred)

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
