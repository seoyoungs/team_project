import db_connect as db
import numpy as np
import pandas as pd
import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # pred = df.iloc[:,1:-1]

    # if is_train == True:
    #     # train, test 나누기
    #     train_value = df[ '2020-09-01' > df['date'] ]
    #     x = train_value.iloc[:,1:-1].astype('float64')
    #     y = train_value.iloc[:,-1].astype('float64')#.to_numpy()
    # else:
    #     test_value = df[df['date'] >=  '2020-09-01']
    #     x = test_value.iloc[:,1:-1].astype('float64')
    #     y = test_value.iloc[:,-1].astype('float64')#.to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    # x = pd.get_dummies(x, columns=["category", "dong"])#.to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return df

df = load_data("SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC")
# x_pred, y_pred = load_data("select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC",is_train=False)


batch_size = 64
trn_loader = torch.utils.data.DataLoader(df,
                                         batch_size=batch_size,
                                         shuffle=True)

print(trn_loader)


