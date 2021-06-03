import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
import tensorflow.keras.backend as K
import scipy.stats as stats
from sklearn.decomposition import PCA

# 0 있다
query = "SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date "
# query1 = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())
# db.cur.execute(query1)
# dataset1 = np.array(db.cur.fetchall())

# pandas 넣기
column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity','person', 'value']

df = pd.DataFrame(dataset, columns=column_name)
# df1 = pd.DataFrame(dataset1, columns=column_name)

db.connect.commit()

train_value = df[ '2020-09-01' > df['date'] ]



x_train1 = train_value.iloc[:,1:].astype('float64')
y_train1 = train_value['value'].astype('float64').to_numpy()

x_train2 = train_value[['temperature','rain','wind','humidity','person']].astype('float64')

from pygam import LinearGAM, s, f
from pygam.datasets import wage
import matplotlib.pyplot as plt
'''
# x_train2, y_train1 = wage()
gam = LinearGAM(s(0) + s(1) + f(2)).fit(x_train2, y_train1)
gam.summary()


for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()
'''


# boston = load_boston()
# df=pd.DataFrame(boston.data, columns=boston.feature_names)

# #使用する変数を選択、正規化
# X = df[['CRIM', 'ZN','RM', 'AGE', 'DIS', 'LSTAT']]
# ss = preprocessing.StandardScaler()
# X_ss = pd.DataFrame(ss.fit_transform(X))
# X_ss.columns=X.columns
# y = boston.target

# #trainとtestに分ける
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# https://rmizutaa.hatenablog.com/entry/2019/03/23/201720

from pygam import LinearGAM
gam = LinearGAM().fit(x_train2, y_train1)

plt.figure(figsize=(12,8))
for i in range(x_train2.shape[1]):
    plt.subplot(2,3,i+1)
    XX = gam.generate_X_grid(term=i)
    plt.plot(XX[:, i],gam.partial_dependence(term=i, X=XX))
    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1],
                c='r', ls='--')
    plt.title(x_train2.columns[i])
plt.show()



