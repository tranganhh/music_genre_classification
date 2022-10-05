from cmath import nan
import csv
from statistics import mean
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

file = r'C:\Users\trang\Desktop\Exemple_projet\music_classification\data\features_30_sec.csv'
reader = pd.read_csv(file, sep=',')
a = reader.iloc[:,2:-1]
b = reader.iloc[:,-1]
b = preprocessing.LabelEncoder().fit_transform(b)
# split data in train en test set
x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.2)
print(a['zero_crossing_rate_mean'])

""" imp = SimpleImputer(missing_values=nan,strategy='mean')
imp = imp.fit(b)
b_imp = imp.transform(b) """



""" a = [1,2,None,3]
imp = SimpleImputer(missing_values=None,strategy='mean')
imp = imp.fit(a)
a_imp = imp.transform(a)
print(a_imp) """
