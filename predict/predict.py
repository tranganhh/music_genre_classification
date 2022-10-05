
from sklearn import metrics
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def extract_test_set(filecsv : str) -> tuple[list, list] : 
    """For extract test set 
    Args:
        filecsv (str): file .csv features extracted

    Returns:
        tuple[list, list]: test set (X - features, y - label)
    """
    file = pd.read_csv(filecsv, sep=',')
    X = file.iloc[:,2:-1]
    transformer = preprocessing.MinMaxScaler()
    x_trans = transformer.fit_transform(X)
    X = pd.DataFrame(x_trans, columns = X.columns)
    Y = file.iloc[:,-1]
    Y = preprocessing.LabelEncoder().fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_test,Y_test

def predict(filecsv : str) -> tuple[list, list]: 
    """For predict

    Args:
        filecsv (str): file .csv features extracted

    Returns:
        tuple[list, list]: y_pred (label predicted), y (label of test set)
    """
    X, y = extract_test_set(filecsv)
    with open('model_classification.pkl', 'rb') as f:
        classifier = pickle.load(f)
    y_pred = classifier.predict(X)
    return (y_pred, y)

def evaluation(filecsv : str) : 
    """Evaluation model

    Args:
        filecsv (str): file .csv features extracted
    """
    y_pred, y = predict(filecsv)
    print(y_pred)
    print(y)
    print("val_accuracy : \n")
    print(metrics.accuracy_score(y, y_pred))



