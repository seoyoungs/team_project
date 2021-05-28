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

# test_value = df1[df1['date'] >=  '2020-09-01']

# x_pred = test_value.iloc[:,1:-1].astype('int64')
# y_pred = test_value['value'].astype('int64').to_numpy()

# x_train1 = pd.get_dummies(x_train1, columns=["category", "dong"]).to_numpy()
# x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()


# 스피어만 상관계수 검정
# corr = x_train1.corr(method='pearson')
# print(corr)

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 분산분석
model = ols('value ~ C(temperature)', x_train1).fit()
print(anova_lm(model))
# temperature : PR(>F) = 0.0
# humidity : PR(>F) = 0.0 
#  가설 : 독립변수(범주형)에 따라 종속변수(연속형)는 유의한 차이가 있다. 
# 귀무가설 : 독립변수(범주형)에 따라 종속변수(연속형)는 유의한 차이가 없다. 
# 대립가설: 독립변수(범주형)에 따라 종속변수(연속형)는 유의한 차이가 있다. 

'''
### ==== 주성분 분석 ===== ###
# feature extraction
pca = PCA(n_components=3)  #PCA 객체 생성
fit = pca.fit(x_train1)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
'''

# from scipy.stats import chi2_contingency
# chi = chi2_contingency(x_train1, correction=False)
# print('chi_square: {}'.format(chi[0]))
# print('p_value: {}'.format(chi[1]))

# import matplotlib.pyplot as plt
# import seaborn as sns
# # 1. 색상구성 결정
# colmap = plt.cm.gist_heat
# # 2. 크기 설정
# plt.figure(figsize=(14,14))
# sns.heatmap(x_train1.corr(),linewidths=0.1,vmax=0.5,cmap=colmap,linecolor='white',annot=True)
# plt.show()

'''
피어슨
                     year         month           day          time      category  ...          rain          wind      humidity        person     value
year         1.000000e+00 -7.820194e-01  3.031852e-03 -1.367524e-15  0.000000e+00  ...  2.955697e-02  1.666402e-01 -2.589314e-03  3.681406e-01  0.009945
month       -7.820194e-01  1.000000e+00 -1.183617e-02 -2.105780e-15  2.215658e-18  ...  4.140563e-02 -1.036942e-01  1.562184e-01 -2.373595e-01 -0.002759
day          3.031852e-03 -1.183617e-02  1.000000e+00  1.159426e-14 -2.217645e-22  ... -2.244798e-02 -3.857282e-02 -3.657656e-02  2.411850e-02  0.003094
time        -1.367524e-15 -2.105780e-15  1.159426e-14  1.000000e+00  0.000000e+00  ... -2.784389e-02  2.161141e-01 -2.425097e-01  2.185163e-15  0.129207
category     0.000000e+00  2.215658e-18 -2.217645e-22  0.000000e+00  1.000000e+00  ... -5.899916e-21 -9.192043e-21 -7.355772e-18 -7.205770e-19 -0.121369
dong         0.000000e+00  9.827526e-17  4.714393e-17  0.000000e+00  0.000000e+00  ... -2.991555e-17 -2.611441e-17  1.747458e-16  4.413861e-17 -0.187172
temperature -7.718446e-02  2.477609e-01  5.396187e-03  1.295938e-01  3.165383e-19  ...  1.169722e-01  4.363108e-03  2.658398e-01 -3.125563e-05  0.022037
rain         2.955697e-02  4.140563e-02 -2.244798e-02 -2.784389e-02 -5.899916e-21  ...  1.000000e+00  9.233988e-02  3.056896e-01  2.789757e-03 -0.000607
wind         1.666402e-01 -1.036942e-01 -3.857282e-02  2.161141e-01 -9.192043e-21  ...  9.233988e-02  1.000000e+00 -2.027154e-01  3.503967e-02  0.048470
humidity    -2.589314e-03  1.562184e-01 -3.657656e-02 -2.425097e-01 -7.355772e-18  ...  3.056896e-01 -2.027154e-01  1.000000e+00  2.376661e-02 -0.044929
person       3.681406e-01 -2.373595e-01  2.411850e-02  2.185163e-15 -7.205770e-19  ...  2.789757e-03  3.503967e-02  2.376661e-02  1.000000e+00  0.009730
value        9.944653e-03 -2.758870e-03  3.094061e-03  1.292067e-01 -1.213689e-01  ... -6.068869e-04  4.847034e-02 -4.492928e-02  9.729553e-03  1.000000

kendall
                 year     month       day      time  category      dong  temperature      rain      wind  humidity    person     value
year         1.000000 -0.686614  0.002649  0.000000  0.000000  0.000000    -0.064672  0.038057  0.136621  0.007884  0.679237  0.005600
month       -0.686614  1.000000 -0.008726  0.000000  0.000000  0.000000     0.176326  0.040484 -0.069429  0.091486 -0.328866  0.003434
day          0.002649 -0.008726  1.000000  0.000000  0.000000  0.000000     0.004632 -0.012866 -0.031093 -0.026316  0.000948  0.003601
time         0.000000  0.000000  0.000000  1.000000  0.000000  0.000000     0.091935 -0.027657  0.164442 -0.164165  0.000000  0.161648
category     0.000000  0.000000  0.000000  0.000000  1.000000  0.000000     0.000000  0.000000  0.000000  0.000000  0.000000 -0.107595
dong         0.000000  0.000000  0.000000  0.000000  0.000000  1.000000     0.000000  0.000000  0.000000  0.000000  0.000000 -0.242607
temperature -0.064672  0.176326  0.004632  0.091935  0.000000  0.000000     1.000000  0.147756  0.013636  0.164869  0.117141  0.029077
rain         0.038057  0.040484 -0.012866 -0.027657  0.000000  0.000000     0.147756  1.000000  0.069313  0.373954  0.052389 -0.009813
wind         0.136621 -0.069429 -0.031093  0.164442  0.000000  0.000000     0.013636  0.069313  1.000000 -0.141819  0.109814  0.070372
humidity     0.007884  0.091486 -0.026316 -0.164165  0.000000  0.000000     0.164869  0.373954 -0.141819  1.000000  0.046278 -0.076931
person       0.679237 -0.328866  0.000948  0.000000  0.000000  0.000000     0.117141  0.052389  0.109814  0.046278  1.000000  0.005276
value        0.005600  0.003434  0.003601  0.161648 -0.107595 -0.242607     0.029077 -0.009813  0.070372 -0.076931  0.005276  1.000000

스피어만
                 year     month       day      time  category      dong  temperature      rain      wind  humidity    person     value
year         1.000000 -0.805917  0.003035  0.000000  0.000000  0.000000    -0.079086  0.039144  0.165237  0.009589  0.768342  0.005716
month       -0.805917  1.000000 -0.011888  0.000000  0.000000  0.000000     0.220093  0.048077 -0.097070  0.136381 -0.449470  0.004097
day          0.003035 -0.011888  1.000000  0.000000  0.000000  0.000000     0.006642 -0.015204 -0.043336 -0.037013  0.000068  0.004200
time         0.000000  0.000000  0.000000  1.000000  0.000000  0.000000     0.136174 -0.034186  0.236791 -0.245142  0.000000  0.197494
category     0.000000  0.000000  0.000000  0.000000  1.000000  0.000000     0.000000  0.000000  0.000000  0.000000  0.000000 -0.129893
dong         0.000000  0.000000  0.000000  0.000000  0.000000  1.000000     0.000000  0.000000  0.000000  0.000000  0.000000 -0.295365
temperature -0.079086  0.220093  0.006642  0.136174  0.000000  0.000000     1.000000  0.187722  0.020288  0.259297  0.157172  0.036301
rain         0.039144  0.048077 -0.015204 -0.034186  0.000000  0.000000     0.187722  1.000000  0.086079  0.460091  0.060585 -0.010301
wind         0.165237 -0.097070 -0.043336  0.236791  0.000000  0.000000     0.020288  0.086079  1.000000 -0.204106  0.150338  0.086856
humidity     0.009589  0.136381 -0.037013 -0.245142  0.000000  0.000000     0.259297  0.460091 -0.204106  1.000000  0.064935 -0.095528
person       0.768342 -0.449470  0.000068  0.000000  0.000000  0.000000     0.157172  0.060585  0.150338  0.064935  1.000000  0.006090
value        0.005716  0.004097  0.004200  0.197494 -0.129893 -0.295365     0.036301 -0.010301  0.086856 -0.095528  0.006090  1.000000

변수 개별 독립성 검정은???
TypeError: unsupported operand type(s) for -: 'float' and 'str'

주성분 분석
Explained Variance: [0.94783679 0.03365549 0.00808704]
이미 앞에서 주성분이 다 되므로 뒤에 변수들이 연관성이 없다(주성분분석이 앞에서 끝남)
[[ 1.65286769e-03 -7.10969723e-03  4.40361360e-04 -1.28660074e-05
  -5.11007989e-08 -1.08630878e-07  1.34411668e-05  4.17708699e-05
   3.79801821e-04  4.55923604e-03  9.99962748e-01  3.11271324e-04]
 [ 5.04144899e-04 -2.89725872e-02  3.47241466e-03  8.44010555e-02
  -3.86834839e-05 -8.61470917e-05 -1.71293620e-01 -2.24370996e-02
   1.17452135e-02 -9.80795738e-01  4.26087509e-03  7.96358333e-03]
 [-4.26402419e-03  7.00682588e-02  3.29439376e-03  2.56169672e-01
  -5.52539549e-04 -1.53141349e-03  9.52580936e-01  3.17031617e-03
   1.19593202e-02 -1.46121045e-01  1.14872593e-03  2.27959864e-02]]

ANOVA
#  가설 : 독립변수(범주형)에 따라 종속변수(연속형)는 유의한 차이가 있다. 
# 귀무가설 : 독립변수(범주형)에 따라 종속변수(연속형)는 유의한 차이가 없다. 
# 대립가설: 독립변수(범주형)에 따라 종속변수(연속형)는 유의한 차이가 있다. 

'''






