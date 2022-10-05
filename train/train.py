
from cmath import nan
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split

def generate_knn_classifier() -> KNeighborsClassifier :
    """Creat KNN classifier

    Returns:
        KNeighborsClassifier: model
    """
    return KNeighborsClassifier(n_neighbors=7, algorithm ='auto', metric =  'manhattan', weights= 'distance')

def extract_training_set(filecsv: str) -> tuple[list, list]:
    """For extract training data set

    Args:
        filecsv (str): file .csv features extracted

    Returns:
        tuple[list, list]: training data set (X - features, y - label)
    """
    file = pd.read_csv(filecsv, sep=',')
    X = file.iloc[:,2:-1]
    transformer = preprocessing.MinMaxScaler()
    x_trans = transformer.fit_transform(X)
    X = pd.DataFrame(x_trans, columns = X.columns)
    Y = file.iloc[:,-1]
    Y = preprocessing.LabelEncoder().fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train,Y_train


def train(filecsv: str, classifier: KNeighborsClassifier) -> KNeighborsClassifier :
    """function for training model with training data

    Args:
        filecsv (str): file .csv features extracted
        classifier (KNeighborsClassifier): model KNN

    Returns:
        KNeighborsClassifier: model KNN after training
    """
    X,y = extract_training_set(filecsv)
    classifier = generate_knn_classifier()
    classifier.fit(X,y)
    print("train_accuracy : \n")
    print(classifier.score(X,y))
    return classifier
    


# file = r'C:\Users\trang\Desktop\Exemple_projet\music_classification\data\features_30_sec.csv'
# #GriSeachCV for knn model
# knn_grid_params = {'n_neighbors': np.arange(3, 50, 2),
#                 'metric':['euclidean', 'manhattan','minkowski'],
#                 'weights':['uniform','distance'],
#                 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
# knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid=knn_grid_params)
# X,y = extract_training_set(filecsv=file)
# knn_grid.fit(X, y)
# best_knn_model = knn_grid.best_estimator_
# print(knn_grid.best_params_)
# print(knn_grid.best_estimator_)
# print("accuracy", knn_grid.best_score_)
